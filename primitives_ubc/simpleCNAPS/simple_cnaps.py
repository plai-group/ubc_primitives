from d3m import container
from d3m.container import pandas # type: ignore
from d3m.base import utils as base_utils
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

# Import config file
from primitives_ubc.config_files import config

import os
import logging
import numpy as np
import torch
from torch.utils import data
from primitives_ubc.simpleCNAPS.dataset   import Dataset
from primitives_ubc.simpleCNAPS.src.model import SimpleCnaps
from primitives_ubc.simpleCNAPS.src.utils import print_and_log, get_log_files
from primitives_ubc.simpleCNAPS.src.utils import loss
from typing import Any, cast, Dict, List, Union, Sequence, Optional, Tuple

__all__ = ('SimpleCNAPSClassifierPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

DEBUG = False  # type: ignore

class Params(params.Params):
    cnaps_model: Optional[Any]

class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        description='Learning rate used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    num_iterations = hyperparams.Hyperparameter[int](
        default=100,
        description="Number of iterations to train the model.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_pretrained = hyperparams.UniformBool(
        default=True,
        description="Whether to use pre-trained model. Set to False to fit on new data.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_two_gpus = hyperparams.UniformBool(
        default=False,
        description="film+ar model does not fit on one GPU, so use 2 GPUs for model parallelism.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    tasks_per_batch= hyperparams.Hyperparameter[int](
        default=16,
        description="Number of tasks per batch",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    feature_adaptation = hyperparams.Enumeration[str](
        values=["no_adaptation", "film", "film+ar"],
        default='film',
        description='Type of activation (non-linearity) following the last layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

class SimpleCNAPSClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Simple CNAPS is a simple classcovariance-based distance metric, namely the Mahalanobis
    distance, adopted into a state-of-the-art few-shot learning approach (CNAPS [https://arxiv.org/abs/1906.07697])
    can, which leads to a significant performance improvement. It is able to learn adaptive
    feature extractors that allow useful estimation of the high dimensional feature covariances required
    by this metric from few samples.
    -------------
    Inputs:  Denormalized dataset.
    Outputs: DataFrame containing the target column of shape Nx1 or Denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    global _weights_configs
    _weights_configs = [{'type': 'FILE',
                         'key': 'pretrained_resnet.pt.tar',
                         'file_uri': 'https://dl.dropboxusercontent.com/s/6ezkjgq2wh9iwwx/pretrained_resnet.pt.tar?dl=1',
                         'file_digest': '347cb3a744a8ff172f7cc47b4b74987f07ca3b6a1f5d6e4f0037474a38e6b285'},
                        {'type': 'FILE',
                         'key': 'best_simple_ar_cnaps.pt',
                         'file_uri': 'https://dl.dropboxusercontent.com/s/4i1xdxqoskp8iuo/best_simple_ar_cnaps.pt?dl=1',
                         'file_digest': '05859f5dec2c70039ff449d37c2474ab07b1a74ab086d2507797d4022d744342'},
                        {'type': 'FILE',
                         'key': 'best_simple_cnaps.pt',
                         'file_uri': 'https://dl.dropboxusercontent.com/s/wk1hoeam4p286oy/best_simple_cnaps.pt?dl=1',
                         'file_digest': '79c93169d567ccb50d4303fbd366560effc4d05dfd7e1a4fa7bca0b3dd0c8d6d'},
    ]
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "166ccbaf-3621-4654-9928-bf7ef17d5c2d",
        "version": config.VERSION,
        "name": "Simple CNAPS primitive",
        "description": "A primitive for few-shot learning classification",
        "python_path": "d3m.primitives.classification.simple_cnaps.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NEURAL_NETWORK_BACKPROPAGATION,
                            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['neural network', 'few-shot learning', 'deep learning'],
        "installation": [config.INSTALLATION] + _weights_configs,
        "hyperparams_to_tune": ['learning_rate']
    })


    def __init__(self, *, hyperparams: Hyperparams, volumes: Union[Dict[str, str], None]=None, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams,  volumes=volumes, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = random_seed
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None
        # Use GPU if available
        use_cuda    = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.use_two_gpus    = self.hyperparams["use_two_gpus"]
        self.tasks_per_batch = self.hyperparams["tasks_per_batch"]
        self.use_pretrained  = self.hyperparams["use_pretrained"]
        # Is the model fit on the training data
        self._fitted = False
        # Arguments
        pretrained_resnet_path = self._find_weights_dir(key_filename="pretrained_resnet.pt.tar", weights_configs=_weights_configs[0])
        self.args = {}
        self.args["feature_adaptation"] = self.hyperparams["feature_adaptation"]
        self.args["pretrained_resnet_path"] = pretrained_resnet_path
        # Setup model
        self._setup_model()

    def _setup_model(self):
        self.model = SimpleCnaps(device=self.device, use_two_gpus=self.use_two_gpus, args=self.args).to(self.device)
        self.model.train() # set encoder is always in train mode to process context data
        self.model.feature_extractor.eval() # feature extractor is always in eval mode
        if self.use_two_gpus:
            self.model.distribute_model()
        self.loss      = loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams["learning_rate"])
        self.optimizer.zero_grad()

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self.use_pretrained:
            self._fitted = True
        else:
            image_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
            base_paths     = [self._training_inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
            base_path      = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))][0] # Path + media

            # Dataset Parameters
            train_params = {'batch_size': 1,
                            'shuffle': False,
                            'num_workers': 0}
            # DataLoader
            training_set = Dataset(datalist=self._training_inputs, base_dir=base_path)

            # Data Generators
            training_generator = data.DataLoader(training_set, **train_params)

            # Number of Iterations
            _iterations = self.hyperparams['num_iterations']

            counter = 0
            for itr in range(_iterations):
                for local_context_images, local_target_images, local_context_labels, local_target_labels in training_generator:
                    local_context_images = torch.squeeze(local_context_images, dim=0).to(self.device)
                    local_context_labels = torch.squeeze(local_context_labels, dim=0).to(self.device)
                    local_target_images  = torch.squeeze(local_target_images,  dim=0).to(self.device)
                    local_target_labels  = torch.squeeze(local_target_labels,  dim=0).to(self.device)
                    # Forward-Pass
                    target_logits = self.model(context_images, context_labels, target_images)
                    task_loss     = self.loss(target_logits, target_labels, self.device) / self.tasks_per_batch
                    if self.args["feature_adaptation"] == 'film' or self.args["feature_adaptation"] == 'film+ar':
                        if self.use_two_gpus:
                            regularization_term = (self.model.feature_adaptation_network.regularization_term()).cuda(0)
                        else:
                            regularization_term = (self.model.feature_adaptation_network.regularization_term())
                        regularizer_scaling = 0.001
                        task_loss += regularizer_scaling * regularization_term
                    task_accuracy  = self.accuracy_fn(target_logits, target_labels)
                    task_loss.backward(retain_graph=False)
                    # Optimize
                    if ((counter+1)%self.tasks_per_batch) or (iteration == (total_iterations - 1)):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
            # Set fit to True
            self._fitted = True

        return base.CallResult(None)


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        # Inference
        if not self._fitted:
            raise Exception('Please fit the model before calling produce!')

        if self.use_pretrained:
            if self.hyperparams["feature_adaptation"] == 'film+ar':
                pretrained_weight_path = self._find_weights_dir(key_filename='best_simple_ar_cnaps.pt', weights_configs=_weights_configs[1])
            else:
                pretrained_weight_path = self._find_weights_dir(key_filename='best_simple_cnaps.pt', weights_configs=_weights_configs[2])
            # Load pre-trained model
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                ckpt_dict = torch.load(pretrained_weight_path)
            else:
                ckpt_dict = torch.load(pretrained_weight_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(ckpt_dict)

        image_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
        base_paths     = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
        base_path      = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))][0] # Path + media

        # Dataset Parameters
        train_params = {'batch_size': 1,
                        'shuffle': False}
        # DataLoader
        testing_set = Dataset(datalist=inputs, base_dir=base_path, mode="TEST")

        # Data Generators
        testing_generator = data.DataLoader(testing_set, **train_params)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(image_columns)

        predictions = []
        with torch.no_grad():
            for local_context_images, local_target_images, local_context_labels, local_target_labels in testing_generator:
                local_context_images = torch.squeeze(local_context_images, dim=0).to(self.device)
                local_context_labels = torch.squeeze(local_context_labels, dim=0).to(self.device)
                local_target_images  = torch.squeeze(local_target_images,  dim=0).to(self.device)
                local_target_labels  = torch.squeeze(local_target_labels,  dim=0).to(self.device)
                # Forwards pass
                target_logits = self.model(local_context_images, local_context_labels, local_target_images)
                averaged_predictions = torch.logsumexp(target_logits,  dim=0)
                final_predictions = torch.argmax(averaged_predictions, dim=-1)
                final_predictions = torch.squeeze(final_predictions)
                final_predictions = final_predictions.data.cpu().numpy()
                # Convert to list
                final_predictions = final_predictions.tolist()
                # Convert context labels to list
                context_labels = torch.squeeze(local_context_labels)
                context_labels = context_labels.data.cpu().numpy()
                context_labels = context_labels.tolist()
                # TODO: add a scoring system for target labels only or edit learningData in LWLL1_metadataset
                predictions.extend(context_labels) # Adding the context labels back
                predictions.extend(final_predictions)

        # Convert from list from DataFrame
        predictions = container.DataFrame(predictions, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(predictions.shape[1]):
            col_dict = dict(predictions.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = 'label'
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
            predictions.metadata        = predictions.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
        # Rename Columns to match label columns similar to LWLL1_metadataset
        predictions.columns = ['label']

        # Append to outputs
        outputs = outputs.append_columns(predictions)

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
        if not self._fitted:
            return Params(cnaps_model=None)

        return Params(cnaps_model=self.model)


    def set_params(self, *, params: Params) -> None:
        self.model = params['cnaps_model']
        self._fitted = True


    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['random_state'] = self._random_state

        return state


    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']
