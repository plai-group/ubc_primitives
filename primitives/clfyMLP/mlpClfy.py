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
from torch.utils import data
import torchvision.transforms as transforms
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple
from primitives.clfyMLP.dataset import Dataset_1
from primitives.clfyMLP.dataset import Dataset_2


__all__ = ('MultilayerPerceptronClassifierPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

DEBUG = False  # type: ignore

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
    use_dropout = hyperparams.UniformBool(
        default=True,
        description="Whether to use dropout after each layer.",
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
        default='softmax',
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
    # Dataset types
    dataset_type = hyperparams.Enumeration[str](
        values=['dataset_1', 'dataset_2'],
        default='dataset_1',
        description='Type of dataset loader to use. dataset_1 when using DataFrame dataset whose Attributes can be converted to NumPy array. dataset_2 when using to read image based dataset',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

class MultilayerPerceptronClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A feed-forward neural network primitive using PyTorch.
    It can be configured with input and output dimensions, number of layers (depth),
    and number of units in each layer except the last one (width).
    -------------
    Inputs:  DataFrame of features/inputs of shape: NxM, where N = samples and M = features/numerical (Attribute) inputs.
             or Denormalized DataFrame of dataset such as image dataset.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "b76a6841-a871-47bc-89ac-e734de5c2924",
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
        #----------------------------------------------------------------------#
        # Final output layer
        if self.hyperparams['last_activation_type'] == 'linear':
            self._last_activation = None
        elif self.hyperparams['last_activation_type'] == 'tanh':
            self._last_activation = nn.Tanh()
        elif self.hyperparams['last_activation_type'] == 'sigmoid':
            self._last_activation = nn.Sigmoid()
        elif self.hyperparams['last_activation_type'] == 'softmax':
            self._last_activation = nn.Softmax(dim=-1)
        else:
            raise ValueError('Unsupported last_activation_type: {}. Available options: linear, tanh, sigmoid, softmax'.format(self.hyperparams['last_activation_type']))
        #----------------------------------------------------------------------#
        # Setup MLP Network
        self._setup_mlp(input_dim=self.hyperparams["input_dim"],\
                        output_dim=self.hyperparams["output_dim"],\
                        depth=self.hyperparams["depth"], width=self.hyperparams["width"],\
                        activation_type=self.hyperparams["activation_type"],\
                        last_activation_type=self._last_activation,\
                        batch_norm=self.hyperparams["use_batch_norm"],\
                        use_dropout=self.hyperparams["use_dropout"])
        #----------------------------------------------------------------------#
        # Pre-processing
        self.pre_process = transforms.Compose([transforms.ToTensor()])
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
                logging.info("\t", name)
                if DEBUG:
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

        # Is the model fit on the training data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def _setup_mlp(self, input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm, use_dropout):
        #----------------------------------------------------------------------#
        if self.hyperparams['output_dim'] < 2:
            raise ValueError("output_dim must be atleast 2!")

        #----------------------------------------------------------------------#
        class _Net(nn.Module):
            def __init__(self, input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm, use_dropout):
                super().__init__()
                self._input_dim = input_dim
                self.activation_type = activation_type
                self.last_activation_type = last_activation_type
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
                self.network = self._make_layers(input_dim, output_dim, depth, width, self._activation, batch_norm=batch_norm, use_dropout=use_dropout)

                # Intialize network
                self._initialize_weights()

            def _make_layers(self, input_dim, output_dim, depth, width, activation, batch_norm, use_dropout):
                layers = []
                in_layers = input_dim
                for v in range(depth):
                    if v == (depth - 1):
                        layer = nn.Linear(in_layers, output_dim)
                        layers += [layer]
                        if use_dropout:
                            layers += [torch.nn.Dropout(p=0.5, inplace=True)]
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
                        if use_dropout:
                            layers += [torch.nn.Dropout(p=0.5, inplace=True)]

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
                    x = self.last_activation_type(x)

                return x

        #----------------------------------------------------------------------#
        self._net = _Net(input_dim, output_dim, depth, width, activation_type, last_activation_type, batch_norm, use_dropout)


    def _curate_data(self, training_inputs, training_outputs, get_labels):
        # if self._training_inputs is None or self._training_outputs is None:
        if training_inputs is None:
            raise ValueError("Missing data.")

        # Get training data and labels data
        try:
            feature_columns_1 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        except:
            feature_columns_1 = None
        try:
            feature_columns_2 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        except:
            feature_columns_2 = None
        # Remove columns if outputs present in inputs
        if len(feature_columns_2) >= 1:
            for fc_2 in feature_columns_2:
                try:
                    feature_columns_1.remove(fc_2)
                except ValueError:
                    pass

        # Get labels data if present in training input
        try:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        except:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # If no error but no label-columns found, force try SuggestedTarget
        if len(label_columns) == 0 or label_columns == None:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # Remove columns if outputs present in inputs
        if len(label_columns) >= 1:
            for lbl_c in label_columns:
                try:
                    feature_columns_1.remove(lbl_c)
                except ValueError:
                    pass

        # Training Set
        feature_columns_1 = [int(fc) for fc in feature_columns_1]
        try:
            new_XTrain = ((training_inputs.iloc[:, feature_columns_1]).to_numpy()).astype(np.float)
        except ValueError:
            # Most likely Numpy ndarray series
            XTrain = training_inputs.iloc[:, feature_columns_1]
            XTrain_shape = XTrain.shape[0]
            XTrain = ((XTrain.iloc[:, -1]).to_numpy())
            # Unpack
            new_XTrain = []
            for arr in range(XTrain_shape):
                new_XTrain.append(XTrain[arr])

            new_XTrain = np.array(new_XTrain)

            # del to save memory
            del XTrain

        # Training labels
        if get_labels:
            if training_outputs is None:
                raise ValueError("Missing data.")

            # Get label column names
            label_name_columns  = []
            label_name_columns_ = list(training_outputs.columns)
            for lbl_c in label_columns:
                label_name_columns.append(label_name_columns_[lbl_c])

            self.label_name_columns = label_name_columns

            # Get labelled dataset
            try:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            except ValueError:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            # If no error but no label-columns force try SuggestedTarget
            if len(label_columns) == 0 or label_columns == None:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            YTrain = ((training_outputs.iloc[:, label_columns]).to_numpy()).astype(np.int)

            return new_XTrain, YTrain, feature_columns_1

        return new_XTrain, feature_columns_1


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self.hyperparams['dataset_type'] == 'dataset_1':
            # Curate data
            XTrain, YTrain, _ = self._curate_data(training_inputs=self._training_inputs, training_outputs=self._training_inputs, get_labels=True)

            # Check if data is matched
            if XTrain.shape[0] != YTrain.shape[0]:
                raise Exception('Size mismatch between training inputs and labels!')

            if YTrain[0].size > 1:
                raise Exception('Primitive accepts labels to be in size (minibatch, 1)!,\
                                 even for multiclass classification problems, it must be in\
                                 the range from 0 to C-1 as the target')

            # Set all files
            _minibatch_size = self.hyperparams['minibatch_size']
            if _minibatch_size > len(XTrain):
                _minibatch_size = len(XTrain)

            # Dataset Parameters
            train_params = {'batch_size': _minibatch_size,
                            'shuffle': self.hyperparams['shuffle'],
                            'num_workers': 4}

            # DataLoader
            training_set = Dataset_1(all_data_X=XTrain, all_data_Y=YTrain, use_labels=True)

            # Data Generators
            training_generator = data.DataLoader(training_set, **train_params)
            #-------------------------------------------------------------------
        else:
            # Get all Nested media files
            image_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
            label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget') # [2]
            if len(label_columns) == 0:
                label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget') # [2]
            base_paths     = [self._training_inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
            base_paths     = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))] # Path + media
            all_img_paths  = [[os.path.join(base_path, filename) for filename in self._training_inputs.iloc[:, col]] for base_path, col in zip(base_paths, image_columns)]
            all_img_labls  = [[os.path.join(label) for label in self._training_outputs.iloc[:, col]] for col in label_columns]

            # Check if data is matched
            for idx in range(len(all_img_paths)):
                if len(all_img_paths[idx]) != len(all_img_labls[idx]):
                    raise Exception('Size mismatch between training inputs and labels!')

            if np.array([all_img_labls[0][0]]).size > 1:
                raise Exception('Primitive accepts labels to be in size (minibatch, 1)!,\
                                 even for multiclass classification problems, it must be in\
                                 the range from 0 to C-1 as the target')

            # Organize data into training format
            all_train_data = []
            for idx in range(len(all_img_paths)):
                img_paths = all_img_paths[idx]
                img_labls = all_img_labls[idx]
                for eachIdx in range(len(img_paths)):
                    all_train_data.append([img_paths[eachIdx], img_labls[eachIdx]])

            # del to free memory
            del all_img_paths, all_img_labls

            if len(all_train_data) == 0:
                raise Exception('Cannot fit when no training data is present.')

            _minibatch_size = self.hyperparams['minibatch_size']
            if _minibatch_size > len(all_train_data):
                _minibatch_size = len(all_train_data)

            # Dataset Parameters
            train_params = {'batch_size': _minibatch_size,
                            'shuffle': self.hyperparams['shuffle'],
                            'num_workers': 4}

            # DataLoader
            training_set = Dataset_2(all_data=all_train_data, preprocess=self.pre_process, use_labels=True)

            # Data Generators
            training_generator = data.DataLoader(training_set, **train_params)
            #-------------------------------------------------------------------

        # Check to add back to class
        self.add_class_index = training_set.sub_class_index

        # Loss function
        if self.hyperparams['loss_type'] == 'crossentropy':
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            raise ValueError('Unsupported loss_type: {}. Available options: crossentropy'.format(self.hyperparams['loss_type']))

        # Train functions
        self._iterations_done = 0
        # Set all files
        _iterations = self.hyperparams['num_iterations']

        # Set model to training
        self._net.train()

        for itr in range(_iterations):
            epoch_loss = 0.0
            iteration  = 0
            for local_batch, local_labels in training_generator:
                # Zero the parameter gradients
                self.optimizer_instance.zero_grad()
                local_batch = torch.flatten(local_batch, start_dim=1)
                # Forward Pass
                local_outputs = self._net(local_batch.to(self.device), inference=False)
                local_labels  = (local_labels.long()).to(self.device)
                # Loss and backward pass
                local_loss = criterion(local_outputs, local_labels)
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
            if DEBUG:
                print('epoch loss: {} at Epoch: {}'.format(epoch_loss, itr))
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
            raise ValueError('Please fit the model before calling produce!')

        if self.hyperparams['dataset_type'] == 'dataset_1':
            # Curate data
            XTest, feature_columns = self._curate_data(training_inputs=inputs, training_outputs=None, get_labels=False)

            # Dataset Parameters
            test_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 4}

            # DataLoader
            testing_set = Dataset_1(all_data_X=XTest, all_data_Y=None, use_labels=False)

            # Data Generators
            testing_generator = data.DataLoader(testing_set, **test_params)
        else:
            # Get all Nested media files
            image_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
            base_paths     = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
            base_paths     = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))] # Path + media
            all_img_paths  = [[os.path.join(base_path, filename) for filename in inputs.iloc[:, col]] for base_path, col in zip(base_paths, image_columns)]

            # Check if data is matched
            for idx in range(len(all_img_paths)):
                if len(all_img_paths[idx]) != len(all_img_labls[idx]):
                    raise Exception('Size mismatch between training inputs and labels!')

            # Organize data into training format
            all_test_data = []
            for idx in range(len(all_img_paths)):
                img_paths = all_img_paths[idx]
                for eachIdx in range(len(img_paths)):
                    all_test_data.append([img_paths[eachIdx]])

            # del to free memory
            del all_img_paths

            if len(all_test_data) == 0:
                raise ValueError('Cannot fit when no training data is present.')

            # DataLoader
            testing_set = Dataset(all_data_X=all_test_data, use_labels=False)

            # Dataset Parameters
            test_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 4}

            # Data Generators
            testing_generator = data.DataLoader(testing_set, **test_params)

            # Using columns
            feature_columns = image_columns
            #-------------------------------------------------------------------

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(feature_columns)

        # Set model to evaluate mode
        self._net.eval()

        predictions = []
        for local_batch in testing_generator:
            local_batch = local_batch.unsqueeze(1) # 1 x F
            local_batch = torch.flatten(local_batch, start_dim=1)
            _out = self._net(local_batch.to(self.device), inference=True)
            _out = torch.argmax(_out, dim=-1, keepdim=False)
            # Add back to predictions if ground truth range is from 1 to C
            if self.add_class_index:
                _out = torch.add(_out, 1)
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
