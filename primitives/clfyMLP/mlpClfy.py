from d3m import container
from d3m.container import pandas # type: ignore
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.base import utils as base_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

from d3m import utils as d3m_utils

# Import config file
from primitives.config_files import config

# Import relevant libraries
import os
import time
import logging
import numpy as np
import pandas as pd
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim # type: ignore
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple
from primitives.clfyMLP.dataset import Dataset


__all__ = ('MultilayerPerceptronClassifierPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    None


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    input_dim = hyperparams.Hyperparameter[int](
        default=100,
        description="Dimensions of the input.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    output_dim = hyperparams.Hyperparameter[int](
        default=2,
        description='Dimensions of CNN output.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    depth = hyperparams.Hyperparameter[int](
        default=2,
        description='Total number of layers, including the output layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    width = hyperparams.Hyperparameter[int](
        default=64,
        description='Number of units in each layer, except the last (output) layer, which is always equal to the output dimensions.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_batch_norm = hyperparams.UniformBool(
        default=False,
        description="Whether to use batch norm after each layer except the last (output) layer.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    activation_type = hyperparams.Enumeration[str](
        values=['linear', 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'softmax'],
        default='relu',
        description='Type of activation (non-linearity) following each layer excet the last one.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    last_activation_type = hyperparams.Enumeration[str](
        values=['linear', 'tanh', 'sigmoid', 'softmax'],
        default='linear',
        description='Type of activation (non-linearity) following the last layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    loss_type = hyperparams.Constant(
        default='crossentropy',
        description='Type of loss used for the local training (fit) of this primitive.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    optimizer_type = hyperparams.Enumeration[str](
        values=['adam', 'sgd'],
        default='adam',
        description='Type of optimizer used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    minibatch_size = hyperparams.Hyperparameter[int](
        default=32,
        description='Minibatch size used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        description='Learning rate used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    momentum = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.9,
        description='Momentum used during training (fit), only for optimizer_type sgd.'
    )
    weight_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.0001,
        description='Weight decay (L2 regularization) used during training (fit).'
    )
    shuffle = hyperparams.UniformBool(
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=True,
        description='Shuffle minibatches in each epoch of training (fit).'
    )
    fit_threshold = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 1e-5,
        description='Threshold of loss value to early stop training (fit).'
    )
    num_iterations = hyperparams.Hyperparameter[int](
        default=100,
        description="Number of iterations to train the model.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class MultilayerPerceptronClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A feed-forward neural network primitive using PyTorch.
    It can be configured with input and output dimensions, number of layers (depth),
    and number of units in each layer except the last one (width).
    -------------
    Inputs:  DataFrame of features/inputs of shape: NxM, where N = samples and M = features/numerical inputs.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "5a7426f9-905e-46ce-945a-3e1fbb67d596",
        "version": config.VERSION,
        "name": "Neural Network Classifier",
        "description": "A feed-forward neural network primitive",
        "python_path": "d3m.primitives.classification.mlp.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NEURAL_NETWORK_BACKPROPAGATION],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['neural network', 'multi-layer Perceptron', 'deep learning'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['learning_rate', 'optimizer_type']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None
        # Use GPU if available
        use_cuda    = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Setup MLP Network
        self._setup_mlp(input_dim=self.hyperparams["input_dim"],\
                        output_dim=self.hyperparams["output_dim"],\
                        depth=self.hyperparams["depth"], width=self.hyperparams["width"],\
                        activation_type=self.hyperparams["activation_type"],\
                        last_activation_type=self.hyperparams["last_activation_type"],\
                        batch_norm=self.hyperparams["use_batch_norm"])

        #----------------------------------------------------------------------#
        # Model to GPU if available
        self._net.to(self.device)

        #----------------------------------------------------------------------#
        # Parameters to update
        self.params_to_update = []
        logging.info("Parameters to learn:")
        for name, param in self._net.named_parameters():
            if param.requires_grad == True:
                self.params_to_update.append(param)
                # logging.info("\t", name)
                print("\t", name)

        #----------------------------------------------------------------------#
        # Optimizer
        if self.hyperparams['optimizer_type'] == 'adam':
            self.optimizer_instance = optim.Adam(self.params_to_update,\
                                             lr=self.hyperparams['learning_rate'],\
                                             weight_decay=self.hyperparams['weight_decay'])
        elif self.hyperparams['optimizer_type'] == 'sgd':
            self.optimizer_instance = optim.SGD(self.params_to_update,\
                                            lr=self.hyperparams['learning_rate'],\
                                            momentum=self.hyperparams['momentum'],\
                                            weight_decay=self.hyperparams['weight_decay'])
        else:
            raise ValueError('Unsupported optimizer_type: {}. Available options: adam, sgd'.format(self.hyperparams['optimizer_type']))

        #----------------------------------------------------------------------#
        # Final output layer
        if self.hyperparams['last_activation_type'] == 'linear':
            self._last_activation = None
        elif self.hyperparams['last_activation_type'] == 'tanh':
            self._last_activation = nn.Tanh()
        elif self.hyperparams['last_activation_type'] == 'sigmoid':
            self._last_activation = nn.Sigmoid()
        elif self.hyperparams['last_activation_type'] == 'softmax':
            self._last_activation = nn.Softmax()
        else:
            raise ValueError('Unsupported last_activation_type: {}. Available options: linear, tanh, sigmoid, softmax'.format(self.hyperparams['last_activation_type']))

        # Is the model fit on the training data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def _setup_mlp(self, input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm):
        #----------------------------------------------------------------------#
        if self.hyperparams['output_dim'] < 2:
            raise ValueError("output_dim must be atleast 2!")

        #----------------------------------------------------------------------#
        class _Net(nn.Module):
            def __init__(self, input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm):
                super().__init__()
                self._input_dim = input_dim
                self.activation_type = activation_type
                # Neuron Activation
                if activation_type == 'linear':
                    self._activation = None
                elif activation_type == 'relu':
                    self._activation = nn.ReLU(inplace=True)
                elif activation_type == 'leaky_relu':
                    self._activation = nn.LeakyReLU(negative_slope=0.01, inplace=False)
                elif activation_type == 'tanh':
                    self._activation = nn.Tanh()
                elif activation_type == 'sigmoid':
                    self._activation = nn.Sigmoid()
                else:
                    raise ValueError('Unsupported activation_type: {}. Available options: linear, relu, tanh, sigmoid'.format(activation_type))

                # Build network
                self.network = self._make_layers(input_dim, output_dim, depth, width, self._activation, batch_norm=batch_norm)

                # Intialize network
                self._initialize_weights()

            def _make_layers(self, input_dim, output_dim, depth, width, activation, batch_norm):
                layers = []
                in_layers = input_dim
                for v in range(depth):
                    if v == (depth - 1):
                        layer = nn.Linear(in_layers, output_dim)
                        layers += [layer]
                    else:
                        layer = nn.Linear(in_layers, width)
                        if activation != None:
                            if batch_norm:
                                layers += [layer, nn.BatchNorm1d(num_features=width), activation]
                            else:
                                layers += [layer, activation]
                        else:
                            if batch_norm:
                                layers += [layer, nn.BatchNorm1d(num_features=width)]
                            else:
                                layers += [layer]
                        in_layers = width

                return nn.Sequential(*layers)

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(self.activation_type))
                        nn.init.constant_(m.bias, 0)

            def forward(self, x, inference=False):
                x = x.view(-1, self._input_dim)
                x = self.network(x)
                if inference:
                    x = self._last_activation(x)

                return x

        #----------------------------------------------------------------------#
        self._net = _Net(input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm)


    def _curate_train_data(self):
        # if self._training_inputs is None or self._training_outputs is None:
        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # Get training data and labels data
        feature_columns = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        # Get labels data if present in training input
        try:
            label_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        except:
            label_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # If no error but no label-columns found, force try SuggestedTarget
        if len(label_columns) == 0 or label_columns == None:
            label_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # Remove columns if outputs present in inputs
        if len(label_columns) >= 1:
            for lbl_c in label_columns:
                try:
                    feature_columns.remove(lbl_c)
                except ValueError:
                    pass

        # Training Set
        feature_columns = [int(fc) for fc in feature_columns]
        XTrain = ((self._training_inputs.iloc[:, feature_columns]).to_numpy()).astype(np.float)

        # Training labels
        try:
            label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        except ValueError:
            label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # If no error but no label-columns force try SuggestedTarget
        if len(label_columns) == 0 or label_columns == None:
            label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        YTrain = ((self._training_outputs.iloc[:, label_columns]).to_numpy()).astype(np.int)
        # Get label column names
        label_name_columns  = []
        label_name_columns_ = list(self._training_outputs.columns)
        for lbl_c in label_columns:
            label_name_columns.append(label_name_columns_[lbl_c])

        self.label_name_columns = label_name_columns

        return XTrain, YTrain


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        # Curate data
        XTrain, YTrain = self._curate_train_data()

        # if self._training_inputs is None or self._training_outputs is None:
        if self.XTrain.shape[1] != self.hyperparams["input_dim"]:
            raise exceptions.InvalidStateError("Training dataset input is not same as input_dim")

        # Check if data is matched
        if self.XTrain.shape[0] != self.YTrain.shape[0]:
            raise Exception('Size mismatch between training inputs and labels!')

        if YTrain[0].size > 1:
            raise Exception('Primitive accepts labels to be in size (minibatch, 1)!,\
                             even for multiclass classification problems, it must be in\
                             the range from 0 to C-1 as the target')

        # Set all files
        _iterations = self.hyperparams['num_iterations']

        _minibatch_size = self.hyperparams['minibatch_size']
        if _minibatch_size > len(all_train_data):
            _minibatch_size = len(all_train_data)

        # Dataset Parameters
        train_params = {'batch_size': _minibatch_size,
                        'shuffle': self.hyperparams['shuffle'],
                        'num_workers': 4}

        # Loss function
        if self.hyperparams['loss_type'] == 'crossentropy':
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            raise ValueError('Unsupported loss_type: {}. Available options: crossentropy'.format(self.hyperparams['loss_type']))

        # Train functions
        self._iterations_done = 0
        # Set all files
        _iterations = self.hyperparams['num_iterations']

        _minibatch_size = self.hyperparams['minibatch_size']
        if _minibatch_size > len(all_train_data):
            _minibatch_size = len(all_train_data)

        # Dataset Parameters
        train_params = {'batch_size': _minibatch_size,
                        'shuffle': self.hyperparams['shuffle'],
                        'num_workers': 4}

        # DataLoader
        training_set = Dataset(all_data_X=XTrain, all_data_Y=YTrain, use_labels=True)

        # Data Generators
        training_generator = data.DataLoader(training_set, **train_params)

        # Set model to training
        self._net.train()

        for itr in range(_iterations):
            epoch_loss = 0.0
            iteration  = 0
            for local_batch, local_labels in training_generator:
                # Zero the parameter gradients
                self.optimizer_instance.zero_grad()
                # Check Label shapes
                if len(local_labels.shape) < 2:
                    local_labels = local_labels.unsqueeze(0)
                # Forward Pass
                local_outputs = self._net(local_batch.to(self.device), inference=False)
                # Loss and backward pass
                local_loss = criterion(local_outputs, local_labels.float())
                local_loss.backward()
                # Update weights
                self.optimizer_instance.step()
                # Increment
                epoch_loss += local_loss
                iteration  += 1
            # Final epoch loss
            epoch_loss /= iteration
            self._iterations_done += 1
            logging.info('epoch loss: {} at Epoch: {}'.format(epoch_loss, itr))
            # print('epoch loss: {} at Epoch: {}'.format(epoch_loss, itr))
            if epoch_loss < self.hyperparams['fit_threshold']:
                self._fitted = True
                return base.CallResult(None)
        self._fitted = True

        return base.CallResult(None)


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        """
        Inputs:  DataFrame of features
        Returns: Pandas DataFrame Containing predictions
        """
        # Inference
        if not self._fitted:
            raise Exception('Please fit the model before calling produce!')

        # Get testing data
        feature_columns = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        # Get labels data if present in testing input
        try:
            label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        except ValueError:
            label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # If no error but no label-columns found, force try SuggestedTarget
        if len(label_columns) == 0 or label_columns == None:
            label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # Remove Label Columns if present from testing data
        if len(label_columns) >= 1:
            for lbl_c in label_columns:
                try:
                    feature_columns.remove(lbl_c)
                except ValueError:
                    pass

        # Testing features
        XTest = ((inputs.iloc[:, feature_columns]).to_numpy()).astype(np.float)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(feature_columns)

        # DataLoader
        testing_set = Dataset(all_data_X=XTest, all_data_Y=None)

        # Dataset Parameters
        test_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 4}

        # Data Generators
        testing_generator = data.DataLoader(testing_set, **test_params)

        # Set model to evaluate mode
        self._net.eval()

        predictions = []
        for local_batch in testing_generator:
            local_batch = local_batch.unsqueeze(0) # 1 x F
            _out = self._net(local_batch.to(self.device), inference=True)
            _out = torch.argmax(_out, dim=-1, keepdim=False)
            _out = _out.data.cpu().numpy()
            # Collect features
            predictions.append(_out)

        # Convert from ndarray from DataFrame
        predictions = container.DataFrame(predictions, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(predictions.shape[1]):
            col_dict = dict(predictions.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = self.label_name_columns[col]
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
            predictions.metadata        = predictions.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
        # Rename Columns to match label columns
        predictions.columns = self.label_name_columns

        # Append to outputs
        outputs = outputs.append_columns(predictions)

        return base.CallResult(outputs)


    def get_params(self) -> Params:
        return None

    def set_params(self, *, params: Params) -> None:
        return None
