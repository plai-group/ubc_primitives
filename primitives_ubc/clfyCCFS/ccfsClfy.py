from d3m import container
from d3m import utils as d3m_utils
from d3m.container import pandas # type: ignore
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive
from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive

# Import config file
from primitives_ubc.config_files import config

# Import relevant libraries
import os
import time
import logging
import scipy.io
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from collections import OrderedDict
from sklearn.impute import SimpleImputer # type: ignore
from typing import Any, cast, Dict, List, Union, Sequence, Optional, Tuple

# Import CCFs functions
from primitives_ubc.clfyCCFS.src.generate_CCF import genCCF
from primitives_ubc.clfyCCFS.src.predict_from_CCF import predictFromCCF

__all__ = ('CanonicalCorrelationForestsClassifierPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    CCF_: Optional[Dict]
    attribute_columns_names: Optional[List[str]]
    target_columns_metadata: Optional[List[OrderedDict]]
    target_columns_names: Optional[List[str]]


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    # Global Hyperparams
    global default_projdict
    default_projdict = OrderedDict()
    default_projdict['CCA'] =  True

    nTrees = hyperparams.UniformInt(
        lower=1,
        upper=10000,
        default=100,
        description="Number of trees to create.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter',
                        'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    )
    parallelprocessing = hyperparams.UniformBool(
        default=True,
        description="Use multi-cpu processing.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    lambda_ = hyperparams.Enumeration[str](
        values=['log', 'sqrt'],
        default='log',
        description="Number of features to subsample at each node",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    splitCriterion = hyperparams.Enumeration[str](
        values=['info', 'gini'],
        default='gini',
        description="Split criterion/impurity measure to use.  Default is 'info' for classification with is entropy impurity/information split criterion.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    minPointsLeaf = hyperparams.Hyperparameter[int](
        default=2,
        description="Minimum number of points allowed a leaf node for split to be permitted.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    bSepPred = hyperparams.UniformBool(
        default=False,
        description="Whether to predict each class seperately as a multilabel classification problem (True) or treat classes within the same output as mutually exclusive (False)",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    taskWeights = hyperparams.Enumeration[str](
        values=['even', 'uneven'], # TODO: Add support for inputing weights list, currently only even supported.
        default='even',
        description="Weights to apply to each output task in calculating the gain.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    bProjBoot = hyperparams.UniformBool(
        default=True,
        description="Whether to use projection bootstrapping.  If set to default, then true unless lambda=D, i.e. we all features at each node.  In this case we resort to bagging instead of projection bootstrapping",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    bBagTrees = hyperparams.UniformBool(
        default=True,
        description="Whether to use Breiman's bagging by training each tree on a bootstrap sample",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    projections = hyperparams.Hyperparameter[dict](
        default=default_projdict,
        description="Whether to use projection bootstrapping.  If set to default, then true unless lambda=D, i.e. we all features at each node.  In this case we resort to bagging instead of projection bootstrapping",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    treeRotation = hyperparams.Enumeration[str](
        values=['none', 'pca', 'random', 'rotationForest'],
        default='none',
        description='Pre-rotation to be applied to each tree seperately before rotating.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    propTrain = hyperparams.Bounded[float](
        lower=0.1,
        upper=1.0,
        default=1.0,
        description="Proportion of the data to train each tree on, but for large datasets it may be possible to only use a subset of the data for training each tree.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    # Numerical stability options. Default values works for most cases
    epsilonCCA = hyperparams.Hyperparameter[float](
        default=1.0000e-04,
        description="Tolerance parameter for rank reduction during the CCA. It can be desirable to lower if the data has extreme correlation, in which this finite value could eliminate the true signal",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    maxDepthSplit = hyperparams.Hyperparameter[str](
        default='stack',
        description="Maximum depth of a node when splitting is still allowed. When set to 'stack' this is set to the maximum value that prevents crashes (usually ~500 which should never really be reached in sensible scenarios)",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    XVariationTol = hyperparams.Hyperparameter[float](
        default=1.0e-10,
        description="Points closer than this tolerance (after scaling the data to unit standard deviation) are considered the same the avoid splitting on numerical error.  Rare would want to change.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    # Options that may want to be set if using algorithms building on CCFs
    RotForM = hyperparams.Hyperparameter[int](
        default=3,
        description="Size of feature subsets taken for each rotation.  Default as per WEKA and rotation forest paper",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    RotForpS = hyperparams.Hyperparameter[float](
        default=0.7500,
        description="Proportion of points to subsample for calculating each PCA projection.  Default as per WEKA but not rotation forest paper",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    RotForpClassLeaveOut = hyperparams.Hyperparameter[float](
        default=0.5000,
        description="Proportion of classes to randomly eliminate for each PCA projection.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    # Properties that can be set but should generally be avoided, using Default works best in most cases.
    minPointsForSplit = hyperparams.Hyperparameter[int](
        default=2,
        description="Minimum points for parent node",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    dirIfEqual = hyperparams.Enumeration[str](
        values=['first', 'rand'],
        default='first',
        description=" When multiple projection vectors can give equivalent split criterion scores, one can either choose which to use randomly ('rand') or take the first ('first') on the basis that the components are in decreasing order of correlation for CCA.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    bContinueProjBootDegenerate = hyperparams.UniformBool(
        default=True,
        description="In the scenario where the projection bootstrap makes the local data pure or have no X variation, the algorithm can either set the node to be a leaf or resort to using the original data for the CCA",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    multiTaskGainCombination = hyperparams.Enumeration[str](
        values=['mean', 'max'],
        default='mean',
        description="Method for combining multiple gain metrics in multi-output tasks. Valid options are 'mean' (default) - average of the gains which for all the considered metrics is equal to the joint gain, or the 'max' gain on any of the tasks.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    missingValuesMethod = hyperparams.Enumeration[str](
        values=['mean', 'random'],
        default='mean',
        description="Method for dealing with missing values.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    # Options that allow nonlinear features to be included in the CCA
    # in accordance with Lopez-Paz's randomized kernel cca.
    bRCCA = hyperparams.UniformBool(
        default=False,
        description="Options that allow nonlinear features to be included in the CCA.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    rccaLengthScale = hyperparams.Hyperparameter[float](
        default=0.1000,
        description="Parameter for bRCCA, if set to True.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    rccaNFeatures = hyperparams.Hyperparameter[int](
        default=6,
        description="Parameter for bRCCA, if set to True.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    rccaRegLambda = hyperparams.Hyperparameter[float](
        default=1.0000e-03,
        description="Parameter for bRCCA, if set to True.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    rccaIncludeOriginal = hyperparams.UniformBool(
        default=False,
        description="Parameter for bRCCA, if set to True.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    # Inputs and outputs HyperParams
    use_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to force primitive to operate on. If any specified column cannot be used, it is skipped.",
    )
    exclude_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    use_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of outputs column indices to force primitive to operate on. If any specified column cannot be used, it is skipped.",
    )
    exclude_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of outputs column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        # Default value depends on the nature of the primitive.
        default='append',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should resulting columns be appended, should they replace original columns, or should only resulting columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no column is selected/provided. Otherwise issue a warning.",
    )


class CanonicalCorrelationForestsClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Canonical Correlation Forests Classifier is a decision tree ensemble method. CCFs naturally
    accommodate multiple outputs, provide a similar computational complexity to random forests,
    and inherit their impressive robustness to the choice of input parameters.
    It uses semantic types to determine which columns to operate on.
    Citation: https://arxiv.org/abs/1507.05444
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame containing the target column of shape Nx1
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "28e8840d-7794-40d2-b5d6-c9e136a6e51e",
        "version": config.VERSION,
        "name": "Canonical Correlation Forests Classifier",
        "description": "A decision tree ensemble primitive like random forests",
        "python_path": "d3m.primitives.classification.canonical_correlation_forests.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DECISION_TREE,\
                            metadata_base.PrimitiveAlgorithmType.ENSEMBLE_LEARNING,\
                            metadata_base.PrimitiveAlgorithmType.CANONICAL_CORRELATION_ANALYSIS,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['canonical correlation forests', 'tree ensemble method', 'decision tree'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['nTrees', 'splitCriterion']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = random_seed
        self._verbose      = _verbose
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        self._CCF = {}
        # Is the model fit on the training data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True
        self._fitted = False


    def _create_learner_param(self) -> None:
        # Setup HyperParams
        self.optionsClassCCF = {}
        self.optionsClassCCF['nTrees']                      = self.hyperparams['nTrees']
        self.optionsClassCCF['parallelprocessing']          = self.hyperparams['parallelprocessing']
        self.optionsClassCCF['lambda']                      = self.hyperparams['lambda_']
        self.optionsClassCCF['splitCriterion']              = self.hyperparams['splitCriterion']
        self.optionsClassCCF['minPointsLeaf']               = self.hyperparams['minPointsLeaf']
        self.optionsClassCCF['bSepPred']                    = self.hyperparams['bSepPred']
        self.optionsClassCCF['taskWeights']                 = self.hyperparams['taskWeights']
        self.optionsClassCCF['bProjBoot']                   = self.hyperparams['bProjBoot']
        self.optionsClassCCF['bBagTrees']                   = self.hyperparams['bBagTrees']
        self.optionsClassCCF['projections']                 = self.hyperparams['projections']
        self.optionsClassCCF['treeRotation']                = self.hyperparams['treeRotation']
        self.optionsClassCCF['propTrain']                   = self.hyperparams['propTrain']
        self.optionsClassCCF['epsilonCCA']                  = self.hyperparams['epsilonCCA']
        self.optionsClassCCF['maxDepthSplit']               = self.hyperparams['maxDepthSplit']
        self.optionsClassCCF['XVariationTol']               = self.hyperparams['XVariationTol']
        self.optionsClassCCF['RotForM']                     = self.hyperparams['RotForM']
        self.optionsClassCCF['RotForpS']                    = self.hyperparams['RotForpS']
        self.optionsClassCCF['RotForpClassLeaveOut']        = self.hyperparams['RotForpClassLeaveOut']
        self.optionsClassCCF['minPointsForSplit']           = self.hyperparams['minPointsForSplit']
        self.optionsClassCCF['dirIfEqual']                  = self.hyperparams['dirIfEqual']
        self.optionsClassCCF['bContinueProjBootDegenerate'] = self.hyperparams['bContinueProjBootDegenerate']
        self.optionsClassCCF['multiTaskGainCombination']    = self.hyperparams['multiTaskGainCombination']
        self.optionsClassCCF['missingValuesMethod']         = self.hyperparams['missingValuesMethod']
        self.optionsClassCCF['bRCCA']                       = self.hyperparams['bRCCA']
        self.optionsClassCCF['rccaLengthScale']             = self.hyperparams['rccaLengthScale']
        self.optionsClassCCF['rccaNFeatures']               = self.hyperparams['rccaNFeatures']
        self.optionsClassCCF['rccaRegLambda']               = self.hyperparams['rccaRegLambda']
        self.optionsClassCCF['rccaIncludeOriginal']         = self.hyperparams['rccaIncludeOriginal']
        self.optionsClassCCF['classNames']                  = np.array([])
        self.optionsClassCCF['org_muY']                     = np.array([])
        self.optionsClassCCF['org_stdY']                    = np.array([])
        self.optionsClassCCF['mseTotal']                    = np.array([])


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")
        self._new_training_data = False

        XTrain, _ = self._select_inputs_columns(self._training_inputs)
        YTrain, _ = self._select_outputs_columns(self._training_outputs)

        print(XTrain)
        print('-----------')
        print(YTrain)

        self._create_learner_param()
        self._store_columns_metadata_and_names(XTrain, YTrain)

        # Fit data
        CCF = genCCF(XTrain, YTrain, nTrees=self.optionsClassCCF['nTrees'], optionsFor=self.optionsClassCCF, do_parallel=self.optionsClassCCF['parallelprocessing'])

        self._CCF    = CCF
        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Inputs:  DataFrame of features
        Returns: Pandas DataFrame Containing predictions
        """
        # Inference
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        XTest, columns_to_use = self._select_inputs_columns(inputs)

        if len(XTest.columns):
            # Prediction
            YpredCCF, _, _  = predictFromCCF(self._CCF, XTest)

            output_columns = [self._wrap_predictions(YpredCCF)]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)


    def get_params(self) -> Params:
        if not self._fitted:
            return Params(CCF_=None,
                          attribute_columns_names=self._attribute_columns_names,
                          target_columns_metadata=self._target_columns_metadata,
                          target_columns_names=self._target_columns_names)

        return Params(CCF_=self._CCF,
                      attribute_columns_names=self._attribute_columns_names,
                      target_columns_metadata=self._target_columns_metadata,
                      target_columns_names=self._target_columns_names)


    def set_params(self, *, params: Params) -> None:
        self._CCF = params['CCF_']
        self._attribute_columns_names = params['attribute_columns_names']
        self._target_columns_metadata = params['target_columns_metadata']
        self._target_columns_names = params['target_columns_names']
        self._fitted = True


    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['random_state'] = self._random_state

        return state


    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']


    def _update_predictions_metadata(self, outputs: Optional[Outputs], target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        outputs_metadata = metadata_base.DataMetadata()
        if outputs is not None:
            outputs_metadata = outputs_metadata.generate(outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata


    def _wrap_predictions(self, predictions: np.ndarray) -> Outputs:
        outputs = container.DataFrame(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(outputs, self._target_columns_metadata)
        outputs.columns = self._target_columns_names
        return outputs


    def _get_target_columns_metadata(self, outputs_metadata: metadata_base.DataMetadata) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = list(column_metadata.get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            semantic_types = [semantic_type for semantic_type in semantic_types if semantic_type != 'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


    def _store_columns_metadata_and_names(self, inputs: Inputs, outputs: Outputs) -> None:
        self._attribute_columns_names = list(inputs.columns)
        self._target_columns_metadata = self._get_target_columns_metadata(outputs.metadata)
        self._target_columns_names = list(outputs.columns)


    def _can_use_inputs_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        return 'https://metadata.datadrivendiscovery.org/types/Attribute' in column_metadata.get('semantic_types', [])


    def _get_inputs_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_inputs_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_inputs_columns'],\
                                                                           self.hyperparams['exclude_inputs_columns'], can_use_column)

        if not columns_to_use:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError("No inputs columns.")
            else:
                self.logger.warning("No inputs columns.")

        if self.hyperparams['use_inputs_columns'] and columns_to_use and columns_not_to_use:
            self.logger.warning("Not all specified inputs columns can be used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use


    def _can_use_outputs_column(self, outputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        return 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in column_metadata.get('semantic_types', [])


    def _get_outputs_columns(self, outputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_outputs_column(outputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(outputs_metadata, self.hyperparams['use_outputs_columns'], self.hyperparams['exclude_outputs_columns'], can_use_column)

        if not columns_to_use:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError("No outputs columns.")
            else:
                self.logger.warning("No outputs columns.")

        if self.hyperparams['use_outputs_columns'] and columns_to_use and columns_not_to_use:
            self.logger.warning("Not all specified outputs columns can be used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use


    def _select_inputs_columns(self, inputs: Inputs) -> Tuple[Inputs, List[int]]:
        columns_to_use = self._get_inputs_columns(inputs.metadata)

        return inputs.select_columns(columns_to_use, allow_empty_columns=True), columns_to_use


    def _select_outputs_columns(self, outputs: Outputs) -> Tuple[Outputs, List[int]]:
        columns_to_use = self._get_outputs_columns(outputs.metadata)

        return outputs.select_columns(columns_to_use, allow_empty_columns=True), columns_to_use
