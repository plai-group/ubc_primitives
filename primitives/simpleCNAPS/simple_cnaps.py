from d3m import container
from d3m.container import pandas # type: ignore
from d3m.base import utils as base_utils
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

# Import config file
from primitives.config_files import config

import logging
import numpy as np
from torch.utils import data
from primitives.simpleCNAPS.dataset import Dataset

__all__ = ('SimpleCNAPSClassifierPrimitive',)
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
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        description='Learning rate used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
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
                         'file_uri': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                         'file_digest': '347cb3a744a8ff172f7cc47b4b74987f07ca3b6a1f5d6e4f0037474a38e6b285'},
                        {'type': 'FILE',
                         'key': 'best_simple_ar_cnaps.pt',
                         'file_uri': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
                         'file_digest': '05859f5dec2c70039ff449d37c2474ab07b1a74ab086d2507797d4022d744342'},
                        {'type': 'FILE',
                         'key': 'best_simple_cnaps.pt',
                         'file_uri': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
                         'file_digest': '79c93169d567ccb50d4303fbd366560effc4d05dfd7e1a4fa7bca0b3dd0c8d6d'},
    ]
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "b76a6841-a871-47bc-89ac-e734de5c2924",
        "version": config.VERSION,
        "name": "Simple CNAPS primitive",
        "description": "A primitive for few-shot learning task",
        "python_path": "d3m.primitives.classification.simpleCnaps.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NEURAL_NETWORK_BACKPROPAGATION],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['neural network', 'few-shot learning', 'deep learning'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['learning_rate']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None
        # Is the model fit on the training data
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        image_columns  = self._training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
        base_paths     = [self._training_inputs.metadata.query((metadata_base.ALL_ELEMENTS, t)) for t in image_columns] # Image Dataset column names
        base_path      = [base_paths[t]['location_base_uris'][0].replace('file:///', '/') for t in range(len(base_paths))][0] # Path + media

        # Dataset Parameters
        train_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 4}
        # DataLoader
        training_set = Dataset(datalist=self._training_inputs, base_dir=base_path)

        # Data Generators
        training_generator = data.DataLoader(training_set, **train_params)

        # print(training_set.__getitem__(index=0))
        # for local_context_images, local_target_images, local_context_labels, local_target_labels in training_generator:
        #     print(local_context_images.shape)
        #     break


        return base.CallResult(None)

    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:

        return 0

    def get_params(self) -> Params:
        return None

    def set_params(self, *, params: Params) -> None:
        return None
