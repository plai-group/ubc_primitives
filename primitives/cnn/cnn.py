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
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple
from primitives.cnn.dataset import Dataset

# Import CNN models
from primitives.cnn.cnn_models.vgg import VGG16
from primitives.cnn.cnn_models.resnet import ResNeT
from primitives.cnn.cnn_models.googlenet import GoogLeNet
from primitives.cnn.cnn_models.mobilenet import MobileNet

__all__ = ('ConvolutionalNeuralNetwork',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class WeightsDirPrimitive:
    """
    Creates a directory for weights if not present in local.
    """
    def __init__(self):
        self._there = False

    @staticmethod
    def _weights_data_dir(dir_name='/static'):
        if not os.path.isdir(dir_name):
            try:
                os.mkdir(dir_name)
            except FileNotFoundError or FileExistsError or PermissionError:
                try:
                    os.mkdir('/static')
                except PermissionError:
                    return None

class Params(params.Params):
    None


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    use_pretrained = hyperparams.UniformBool(
        default=True,
        description="Whether to use pre-trained ImageNet weights",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    train_endToend = hyperparams.UniformBool(
        default=False,
        description="Whether to train the network end to end or fine-tune the last layer only.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    use_batch_norm = hyperparams.UniformBool(
        default=False,
        description="Whether to use batch norm for VGG network",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    feature_extract_only = hyperparams.UniformBool(
        default=True,
        description="Whether to use CNN as feature extraction only without training",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    include_top = hyperparams.UniformBool(
        default=True,
        description="Whether to use top layers, i.e. final fully connected layers",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    img_resize = hyperparams.Hyperparameter[int](
        default=224,
        description="Size to resize the input image",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    output_dim = hyperparams.Hyperparameter[int](
        default=1000,
        description='Dimensions of CNN output.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    last_activation_type = hyperparams.Enumeration[str](
        values=['linear', 'relu', 'tanh', 'sigmoid', 'softmax'],
        default='linear',
        description='Type of activation (non-linearity) following the last layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    cnn_type = hyperparams.Enumeration[str](
        values=['vgg', 'googlenet', 'mobilenet', 'resnet'],
        default='resnet',
        description='Type of convolutional neural network to use.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    loss_type = hyperparams.Enumeration[str](
        values=['mse', 'crossentropy', 'l1'],
        default='mse',
        description='Type of loss used for the local training (fit) of this primitive.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
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


class ConvolutionalNeuralNetwork(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams], WeightsDirPrimitive):
    """
    Convolutional Neural Network primitive using PyTorch framework.
    Used to extract deep features from images.
    It can be used as a pre-trained feature extractor, to extract features from
    convolutional layers or the fully connected layers by setting include_top.
    It can also be fine-tunned to fit new data, by setting feature extraction to False.
    Available pre-trained CNN models are:
      - VGG-16
      - VGG-16 with Batch-Norm
      - GoogLeNet
      - ResNeT
      - MobileNet (A Light weight CNN model)
    All available models are pre-trained on ImageNet.
    """
    # Check if the weights directory exist, else create one. Default: /static
    WeightsDirPrimitive._weights_data_dir(dir_name="/ubc_primitives/static")
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    global _weights_configs
    _weights_configs = [{'type': 'FILE',
                         'key': 'vgg16-397923af.pth',
                         'file_uri': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                         'file_digest': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'},
                         {'type': 'FILE',
                          'key': 'vgg16_bn-6c64b313.pth',
                          'file_uri': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
                          'file_digest': '6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a'},
                         {'type': 'FILE',
                          'key': 'googlenet-1378be20.pth',
                          'file_uri': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
                          'file_digest': '1378be20a8e875cf1568b8a71654e704449655e34711a959a38b04fb34905cef'},
                         {'type': 'FILE',
                          'key': 'mobilenet_v2-b0353104.pth',
                          'file_uri': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                          'file_digest': 'b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889'},
                         {'type': 'FILE',
                          'key': 'resnet34-333f7ec4.pth',
                          'file_uri': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                          'file_digest': '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'},
    ]
    metadata = metadata_base.PrimitiveMetadata({
        "id": "88152884-dc0c-40e5-ba07-6a6c9cd45ef1",
        "version": config.VERSION,
        "name": "Convolutional Neural Network",
        "description": "A primitive to extract features and to fit model for images",
        "python_path": "d3m.primitives.feature_extraction.cnn.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ["cnn", "vgg", "googlenet", "resnet", "mobilenet", "convolutional neural network", "deep learning"],
        "installation": [config.INSTALLATION] + _weights_configs,
        "hyperparams_to_tune": ['learning_rate', 'cnn_type']
    })


    def __init__(self, *, hyperparams: Hyperparams, volumes: Union[Dict[str, str], None]=None):
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        self.hyperparams = hyperparams
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        # Use GPU if available
        use_cuda    = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        # Setup Convolutional Network
        self._setup_cnn()
        # Image pre-processing function
        self._img_size = int(self.hyperparams['img_resize'])
        if self.hyperparams['cnn_type'] == 'googlenet':
            # Normalize done inside GoogLeNet model
            self.val_pre_process = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(self._img_size),
                                    transforms.ToTensor()])
            # Random Crop during training
            self.pre_process = transforms.Compose([
                                transforms.Resize(255),
                                transforms.RandomCrop(self._img_size),
                                transforms.ToTensor()])
        else:
            # All other pre-trained models are normalized in the same way
            self.val_pre_process = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(self._img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                         std=[0.229, 0.224, 0.225])])
            # Random Crop during training
            self.pre_process = transforms.Compose([
                                transforms.Resize(255),
                                transforms.RandomCrop(self._img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])])
        # Is the model fit on data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def _setup_cnn(self):
        #----------------------------------------------------------------------#
        if self.hyperparams['feature_extract_only'] and self.hyperparams['include_top']:
            self.include_last_layer = False
        else:
            self.include_last_layer = True

        #--------------------------------VGG-----------------------------------#
        if self.hyperparams['cnn_type'] == 'vgg':
            # Get CNN Model
            self.model = VGG16(include_top=self.hyperparams['include_top'])
            if self.hyperparams['use_pretrained']:
                if self.hyperparams['use_batch_norm']:
                    weights_path = self._find_weights_dir(key_filename='vgg16_bn-6c64b313.pth', weights_configs=_weights_configs[1])
                else:
                    weights_path = self._find_weights_dir(key_filename='vgg16-397923af.pth', weights_configs=_weights_configs[0])
                checkpoint = torch.load(weights_path)
                self.model.load_state_dict(checkpoint)
                self.expected_feature_out_dim = (512 * 7 * 7)
                #logging.info("Pre-Trained imagenet weights loaded!")
                print("Pre-Trained imagenet weights loaded!")

            # Final layer Augmentation
            if (not self.hyperparams['feature_extract_only']) and self.hyperparams['output_dim'] != 1000:
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_ftrs, self.hyperparams['output_dim'])
                # Intialize with random weights
                nn.init.normal_(self.model.classifier[6].weight, 0, 0.01)
                nn.init.constant_(self.model.classifier[6].bias, 0)

            # Remove final layer if needed
            if (not self.include_last_layer) and self.hyperparams['feature_extract_only']:
                clsfy_layers = self.model.classifier
                self.model.classifier = nn.Sequential( *list(clsfy_layers)[:-1])

            # Freeze all layers except the last layer and gather params
            if not self.hyperparams['train_endToend']:
                for param in self.model.parameters():
                    param.requires_grad = False
                if (not self.hyperparams['feature_extract_only']):
                    for parameter in self.model.classifier[6].parameters():
                        parameter.requires_grad = True

        #----------------------------GoogLeNet---------------------------------#
        elif self.hyperparams['cnn_type'] == 'googlenet':
            # Get CNN Model
            self.model = GoogLeNet(include_top=self.hyperparams['include_top'], num_classes=int(self.hyperparams['output_dim']))
            if self.hyperparams['use_pretrained']:
                weights_path = self._find_weights_dir(key_filename='googlenet-1378be20.pth', weights_configs=_weights_configs[2])
                checkpoint   = torch.load(weights_path)
                self.model.load_state_dict(checkpoint)
                self.expected_feature_out_dim = (1024 * 7 * 7)
                logging.info("Pre-Trained imagenet weights loaded!")

            # Final layer Augmentation
            if (not self.hyperparams['feature_extract_only']) and self.hyperparams['output_dim'] != 1000:
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, self.hyperparams['output_dim'])
                # Intialize with random weights
                nn.init.normal_(self.model.fc.weight, 0, 0.01)
                nn.init.constant_(self.model.fc.bias, 0)

            # Freeze all layers except the last layer and gather params
            if not self.hyperparams['train_endToend']:
                for param in self.model.parameters():
                    param.requires_grad = False
                if (not self.hyperparams['feature_extract_only']):
                    for parameter in self.model.fc.parameters():
                        parameter.requires_grad = True

        #----------------------------MobileNet---------------------------------#
        elif self.hyperparams['cnn_type'] == 'mobilenet':
            # Get CNN Model
            self.model = MobileNet(include_top=self.hyperparams['include_top'])
            if self.hyperparams['use_pretrained']:
                weights_path = self._find_weights_dir(key_filename='mobilenet_v2-b0353104.pth', weights_configs=_weights_configs[3])
                checkpoint   = torch.load(weights_path)
                self.model.load_state_dict(checkpoint)
                self.expected_feature_out_dim = (1280 * 7 * 7)
                logging.info("Pre-Trained imagenet weights loaded!")

            # Final layer Augmentation
            if (not self.hyperparams['feature_extract_only']) and self.hyperparams['output_dim'] != 1000:
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_ftrs, self.hyperparams['output_dim'])
                # Intialize with random weights
                nn.init.normal_(self.model.classifier[1].weight, 0, 0.01)
                nn.init.constant_(self.model.classifier[1].bias, 0)

            # Freeze all layers except the last layer and gather params
            if not self.hyperparams['train_endToend']:
                for param in self.model.parameters():
                    param.requires_grad = False
                if (not self.hyperparams['feature_extract_only']):
                    for parameter in self.model.classifier[1].parameters():
                        parameter.requires_grad = True

        #-----------------------------ResNeT-----------------------------------#
        elif self.hyperparams['cnn_type'] == 'resnet':
            # Get CNN Model
            self.model = ResNeT(include_top=self.hyperparams['include_top'])
            if self.hyperparams['use_pretrained']:
                weights_path = self._find_weights_dir(key_filename='resnet34-333f7ec4.pth', weights_configs=_weights_configs[4])
                checkpoint   = torch.load(weights_path)
                self.model.load_state_dict(checkpoint)
                self.expected_feature_out_dim = (512 * 7 * 7)
                logging.info("Pre-Trained imagenet weights loaded!")

            # Final layer Augmentation
            if (not self.hyperparams['feature_extract_only']) and self.hyperparams['output_dim'] != 1000:
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, self.hyperparams['output_dim'])
                # Intialize with random weights
                nn.init.normal_(self.model.fc.weight, 0, 0.01)
                nn.init.constant_(self.model.fc.bias, 0)

            # Freeze all layers except the last layer and gather params
            if not self.hyperparams['train_endToend']:
                for param in self.model.parameters():
                    param.requires_grad = False
                if (not self.hyperparams['feature_extract_only']):
                    for parameter in self.model.fc.parameters():
                        parameter.requires_grad = True

        #----------------------------------------------------------------------#
        # Model to GPU if available
        self.model.to(self.device)
        # print(self.model)

        #----------------------------------------------------------------------#
        # Parameters to update
        self.params_to_update = []
        if (not self.hyperparams['feature_extract_only']):
            logging.info("Parameters to learn:")
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    logging.info("\t", name)

        #----------------------------------------------------------------------#
        # Optimizer
        if (not self.hyperparams['feature_extract_only']):
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
        if self.hyperparams['last_activation_type']   == 'relu':
            self.final_layer = nn.ReLU(inplace=True).to(self.device)
        elif self.hyperparams['last_activation_type'] == 'tanh':
            self.final_layer = nn.Tanh().to(self.device)
        elif self.hyperparams['last_activation_type'] == 'sigmoid':
            self.final_layer = nn.Sigmoid().to(self.device)
        elif self.hyperparams['last_activation_type'] == 'softmax':
            self.final_layer = nn.Softmax(dim=1).to(self.device)
        else:
            self.final_layer = None
        #----------------------------------------------------------------------#

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Inputs: Dataset dataFrame
        Returns: None
        """
        # If feature extract only, Skip Fit
        if self.hyperparams['feature_extract_only']:
            self._fitted = True
            return base.CallResult(None)

        if self._fitted:
            return base.CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

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
        training_set = Dataset(all_data=all_train_data, preprocess=self.pre_process)

        # Data Generators
        training_generator = data.DataLoader(training_set, **train_params)

        # Set model to training mode
        self.model.train()

        # Loss function
        if self.hyperparams['loss_type'] == 'crossentropy':
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.hyperparams['loss_type'] == 'mse':
            criterion = nn.MSELoss().to(self.device)
        elif self.hyperparams['loss_type'] == 'l1':
            criterion = nn.L1Loss().to(self.device)
        else:
            raise ValueError('Unsupported loss_type: {}. Available options: crossentropy, mse, l1'.format(self.hyperparams['loss_type']))

        # Train functions
        start = time.time()
        self._iterations_done = 0

        # Set model to training
        self.model.train()

        for itr in range(_iterations):
            epoch_loss = 0.0
            iteration  = 0
            for local_batch, local_labels in training_generator:
                # Zero the parameter gradients
                self.optimizer_instance.zero_grad()
                # Check Label shapes
                if len(local_labels.shape) < 2:
                    local_labels = local_labels.unsqueeze(1)
                elif len(local_labels.shape) > 2:
                    raise Exception('Primitive accepts labels to be in size (minibatch, 1)!,\
                                     even for multiclass classification problems, it must be in\
                                     the range from 0 to C-1 as the target')
                # Forward Pass
                if self.hyperparams['cnn_type'] == 'googlenet':
                    local_outputs, aux_1, aux_2 = self.model(local_batch.to(self.device), include_last_layer=self.include_last_layer)
                    if self.hyperparams['train_endToend']:
                        # Loss
                        local_loss = criterion(local_outputs, local_labels.float())
                        local_loss += 0.4 * criterion(aux_1,  local_labels.float())
                        local_loss += 0.4 * criterion(aux_2,  local_labels.float())
                    else:
                        local_loss = criterion(local_outputs, local_labels.float())
                    # Backward pass
                    local_loss.backward()
                else:
                    local_outputs = self.model(local_batch.to(self.device), include_last_layer=self.include_last_layer)
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
        Inputs: Dataset dataFrame
        Returns: Pandas DataFramefor for classification or regression task
        """
        # Get all Nested media files
        image_columns = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
        base_paths    = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
        base_paths    = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))] # Path + media
        all_img_paths = [[os.path.join(base_path, filename) for filename in inputs.iloc[:,col]] for base_path, col in zip(base_paths, image_columns)]

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(image_columns)

        # Set model to evaluate mode
        self.model.eval()

        return_argmax = False
        if self.hyperparams['loss_type'] == 'crossentropy':
            # Multi-class classification, therefore call argmax on inference
            return_argmax = True

        # Feature extraction without fitting
        if self.hyperparams['feature_extract_only']:
            features = []
            for idx in range(len(all_img_paths)):
                img_paths = all_img_paths[idx]
                for imagefile in img_paths:
                    if os.path.isfile(imagefile):
                        image = Image.open(imagefile)
                        image = self.val_pre_process(image) # To pytorch tensor
                        image = image.unsqueeze(0) # 1 x C x H x W
                        if self.hyperparams['cnn_type'] == 'googlenet':
                            feature, _, _ = self.model(image.to(self.device), include_last_layer=self.include_last_layer)
                        else:
                            feature = self.model(image.to(self.device), include_last_layer=self.include_last_layer)
                        if self.final_layer != None:
                            feature = self.final_layer(feature)
                        if len(feature.shape) > 1:
                            feature = torch.flatten(feature)
                        feature = feature.data.cpu().numpy()
                        #print(feature.shape)
                    else:
                        logging.warning("No such file {}. Feature vector will be set to all zeros.".format(file_path))
                        feature = np.zeros((self.expected_feature_out_dim))
                    # Collect features
                    features.append(feature)
            # Feature vector data frame
            feature_vectors = container.DataFrame(features, generate_metadata=True)

            # Update Metadata for each feature vector dataframe column
            for col in range(feature_vectors.shape[1]):
                col_dict = dict(feature_vectors.metadata.query((metadata_base.ALL_ELEMENTS, col)))
                col_dict['structural_type'] = type(1.0)
                col_dict['name']            = "vector_" + str(col)
                col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
                feature_vectors.metadata    = feature_vectors.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

            # Add the features to the input labels with data removed
            outputs = outputs.append_columns(feature_vectors)
        #-----------------------------------------------------------------------
        else:
            # Inference
            if not self._fitted and self.hyperparams['output_dim'] != 1000:
                raise Exception('Please fit the model before calling produce!')

            # Get Label Columns Names
            label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget') # [2]
            if len(label_columns) == 0 or label_columns == None:
                label_columns  = self._training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget') # [2]
            label_columns_names = [list(self._training_outputs.columns)[i] for i in label_columns]

            predictions = []
            for idx in range(len(all_img_paths)):
                img_paths = all_img_paths[idx]
                for imagefile in img_paths:
                    if os.path.isfile(imagefile):
                        image = Image.open(imagefile)
                        image = self.val_pre_process(image) # To pytorch tensor
                        image = image.unsqueeze(0) # 1 x C x H x W
                        if self.hyperparams['cnn_type'] == 'googlenet':
                            _out, _, _ = self.model(image.to(self.device), include_last_layer=self.include_last_layer)
                        else:
                            _out  = self.model(image.to(self.device), include_last_layer=self.include_last_layer)
                        if return_argmax:
                            _out = torch.argmax(_out, dim=-1, keepdim=False)
                        _out  = torch.flatten(_out)
                        _out  = _out.data.cpu().numpy()
                    else:
                        logging.warning("No such file {}. Feature vector will be set to all zeros.".format(file_path))
                        _out = np.zeros((self.hyperparams['output_dim']))
                    # Collect features
                    predictions.append(_out)

            # Convert to d3m type with metadata
            preds = container.DataFrame(predictions, generate_metadata=True)

            # Update Metadata for each feature vector column
            for col in range(preds.shape[1]):
                col_dict = dict(preds.metadata.query((metadata_base.ALL_ELEMENTS, col)))
                col_dict['structural_type'] = type(1.0)
                col_dict['name']            = label_columns_names[col]
                col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
                preds.metadata = preds.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

            # Add the features to the input labels with data removed
            outputs = outputs.append_columns(preds)
        #-----------------------------------------------------------------------

        return base.CallResult(outputs)


    def _find_weights_dir(self, key_filename, weights_configs):
        _weight_file_path = None
        # Check common places
        if key_filename in self.volumes:
            _weight_file_path = self.volumes[key_filename]
        else:
            if os.path.isdir('/static'):
                _weight_file_path = os.path.join('/static', weights_configs['file_digest'], key_filename)
                if not os.path.exists(_weight_file_path):
                    _weight_file_path = os.path.join('/static', weights_configs['file_digest'])
            # Check other directories
            if not os.path.exists(_weight_file_path):
                home = expanduser("/")
                root = expanduser("~")
                _weight_file_path = os.path.join(home, weights_configs['file_digest'])
                if not os.path.exists(_weight_file_path):
                    _weight_file_path = os.path.join(home, weights_configs['file_digest'], key_filename)
                if not os.path.exists(_weight_file_path):
                    _weight_file_path = os.path.join('.', weights_configs['file_digest'], key_filename)
                if not os.path.exists(_weight_file_path):
                    _weight_file_path = os.path.join(root, weights_configs['file_digest'], key_filename)
                if not os.path.exists(_weight_file_path):
                    _weight_file_path = os.path.join(weights_configs['file_digest'], key_filename)

        if os.path.isfile(_weight_file_path):
            return _weight_file_path
        else:
            raise ValueError("Can't get weights file from the volume by key: {} or in the static folder: {}".format(key_filename, _weight_file_path))

        return _weight_file_path


    def get_params(self) -> Params:
        return None

    def set_params(self, *, params: Params) -> None:
        return None
