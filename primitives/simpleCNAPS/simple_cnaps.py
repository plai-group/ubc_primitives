from d3m import container
from d3m.container import pandas # type: ignore
from d3m.base import utils as base_utils
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

__all__ = ('SimpleCnapsClassifierPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

class SimpleCnapsClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
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
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "b76a6841-a871-47bc-89ac-e734de5c2924",
        "version": config.VERSION,
        "name": "Simple CNAPS",
        "description": "A few shot learning",
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
        "hyperparams_to_tune": ['learning_rate', 'optimizer_type']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None
