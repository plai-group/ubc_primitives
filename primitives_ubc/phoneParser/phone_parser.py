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
import pyprob
from torch.utils import data
from typing import Any, cast, Dict, List, Union, Sequence, Optional, Tuple

from infcomp import PhoneParser

__all__ = ('PhoneNumberParserPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

DEBUG = False  # type: ignore


class Params(params.Params):
    parser_model: Optional[Any]


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    num_samples = hyperparams.Hyperparameter[int](
        default=10,
        description="Samples to sample from the posterior.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=128,
        description="Batch size for training.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    eval_num_traces = hyperparams.Hyperparameter[int](
        default=10,
        description='Traces to evaluate during inference.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    train_num_traces = hyperparams.Hyperparameter[int](
        default=5000000,
        description='Traces to evaluate per training step.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_pretrained = hyperparams.UniformBool(
        default=True,
        description="Whether to use pre-trained ImageNet weights",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class PhoneNumberParserPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Parsing using inference compilation
    -------------
    Inputs:  Denormalized dataset.
    Outputs: DataFrame containing the target column of shape Nx1 or Denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    global _weights_configs
    _weights_configs = [{'type': 'FILE',
                         'key': 'phone_parser',
                         'file_uri': 'https://dl.dropboxusercontent.com/s/6ezkjgq2wh9iwwx/pretrained_resnet.pt.tar?dl=1',
                         'file_digest': '51872ef411d1f34fa9986ecebaeac2137d3f05a9aec2bac2a0544bce57c66237'},
    ]
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "166ccbaf-3621-4654-9928-bf7ef17d5c2d",
        "version": config.VERSION,
        "name": "Phone parser primitive",
        "description": "A primitive to parser international phone numbers",
        "python_path": "d3m.primitives.classification.phone_number.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NEURAL_NETWORK_BACKPROPAGATION],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['neural network', 'phone parser', 'pyprob', 'inference'],
        "installation": [config.INSTALLATION] + _weights_configs,
        "hyperparams_to_tune": ['batch_size', 'train_num_traces']
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
        self.batch_size  = self.hyperparams["batch_size"]
        self.num_samples = self.hyperparams["num_samples"]
        self.eval_num_traces  = self.hyperparams["eval_num_traces"]
        self.train_num_traces = self.hyperparams["train_num_traces"]
        self.use_pretrained   = self.hyperparams["use_pretrained"]

        # Setup model
        self.model = PhoneParser()

        # Is the model fit on the training data
        self._fitted = False

        # Arguments
        self.model = PhoneParser()

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Inputs: Dataset dataFrame
        Returns: None
        """
        if self.use_pretrained:
            self._fitted = True
            return base.CallResult(None)

        if self._fitted:
            return base.CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        self.model.learn_inference_network(
            inference_network=pyprob.InferenceNetwork.LSTM,
            observe_embeddings={'phone_string': {'dim' : 256}},
            num_traces=self.train_num_traces,
            batch_size=self.batch_size,
            save_file_name_prefix=MODEL_PATH,
        )

        return base.CallResult(None)


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        """
        Inputs:  Dataset dataFrame
        Returns: Pandas DataFrame
        """
        if self.use_pretrained:
            # Use pre-trained
            model.load_inference_network(MODEL_PATH)

        post = model.posterior_distribution(
            observe=model.get_observes(NUMBER),
            inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
            num_traces=NUM_TRACES
        )


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
            return Params(parser_model=None)

        return Params(parser_model=self.model)


    def set_params(self, *, params: Params) -> None:
        self.model = params['parser_model']
        self._fitted = True


    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['random_state'] = self._random_state

        return state


    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']
