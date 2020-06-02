from d3m import container
from d3m.container import pandas # type: ignore
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.base import utils as base_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive
from common_primitives.ndarray_to_dataframe import NDArrayToDataFramePrimitive

from d3m import utils as d3m_utils

# Import config file
from primitives_ubc.config_files import config

# Import relevant libraries
import os
import time
import logging
import numpy as np
from collections  import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple

# Import CCFs functions
from primitives_ubc.regCCFS.src.generate_CCF import genCCF
from primitives_ubc.regCCFS.src.predict_from_CCF import predictFromCCF

__all__ = ('CanonicalCorrelationForestsRegressionPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    None


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    # Global Hyperparams
    global default_projdict
    default_projdict = OrderedDict()
    default_projdict['CCA'] =  True

    nTrees = hyperparams.Hyperparameter[int](
        default=100,
        description="Number of trees to create.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
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
        values=['mse'],
        default='mse',
        description="Split criterion/impurity measure to use.  Default is 'mse' for Regression.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    minPointsLeaf = hyperparams.Hyperparameter[int](
        default=3,
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
    mseErrorTolerance = hyperparams.Hyperparameter[float](
        default=1e-6,
        description=" When doing regression with mse splits, the node is made into a leaf if the mse (i.e. variance) of the data is less  than this tolerance times the mse of the full data set.",
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
    # Properties that can be set but should generally be avoided, use Default works best.
    minPointsForSplit = hyperparams.Hyperparameter[int](
        default=6,
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
        default='random',
        description="Method for dealing with missing values.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    bUseOutputComponentsMSE = hyperparams.UniformBool(
        default=False,
        description="If true, doing regression with multiple outputs and doing CCA projections.",
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
        default=50,
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



class CanonicalCorrelationForestsRegressionPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Canonical Correlation Forests regression is a decision tree ensemble method. CCFs naturally
    accommodate multiple outputs, provide a similar computational complexity to random forests,
    and inherit their impressive robustness to the choice of input parameters.
    It uses semantic types to determine which columns to operate on.
    Citation: https://arxiv.org/abs/1507.05444
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "422c040c-b8fe-45de-89c2-01d17118379d",
        "version": config.VERSION,
        "name": "Canonical Correlation Forests Regressor",
        "description": "A decision tree ensemble primitive like random forests",
        "python_path": "d3m.primitives.regression.ccfs.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
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
        "hyperparams_to_tune": ['nTrees']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        # Is the model fit on the training data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def _create_learner_param(self):
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
        self.optionsClassCCF['mseErrorTolerance']           = self.hyperparams['mseErrorTolerance']
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
        self.optionsClassCCF['bUseOutputComponentsMSE']     = self.hyperparams['bUseOutputComponentsMSE']
        self.optionsClassCCF['bRCCA']                       = self.hyperparams['bRCCA']
        self.optionsClassCCF['rccaLengthScale']             = self.hyperparams['rccaLengthScale']
        self.optionsClassCCF['rccaNFeatures']               = self.hyperparams['rccaNFeatures']
        self.optionsClassCCF['rccaRegLambda']               = self.hyperparams['rccaRegLambda']
        self.optionsClassCCF['rccaIncludeOriginal']         = self.hyperparams['rccaIncludeOriginal']
        self.optionsClassCCF['classNames']                  = np.array([])
        self.optionsClassCCF['org_muY']                     = np.array([])
        self.optionsClassCCF['org_stdY']                    = np.array([])
        self.optionsClassCCF['mseTotal']                    = np.array([])


    def _curate_train_data(self):
        """
        Process DataFrame and convert to Numpy array
        """
        # if self._training_inputs is None or self._training_outputs is None:
        if self._training_inputs is None:
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
        YTrain = ((self._training_outputs.iloc[:, label_columns]).to_numpy()).astype(np.float)
        # Get label column names
        label_name_columns  = []
        label_name_columns_ = list(self._training_outputs.columns)
        for lbl_c in label_columns:
            label_name_columns.append(label_name_columns_[lbl_c])

        self.label_name_columns = label_name_columns

        return XTrain, YTrain

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Inputs: ndarray of features
        Returns: None
        """
        self._create_learner_param()
        XTrain, YTrain = self._curate_train_data()

        # Fit data
        CCF = genCCF(XTrain, YTrain, nTrees=self.optionsClassCCF['nTrees'], bReg=True, optionsFor=self.optionsClassCCF, do_parallel=self.optionsClassCCF['parallelprocessing'])

        self.CCF = CCF
        self._fitted = True

        return base.CallResult(None)


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        """
        Inputs:  ndarray of features
        Returns: Pandas DataFrame Containing predictions
        """
        # Inference
        if not self._fitted:
            raise Exception('Please fit the model before calling produce!')

        # Get testing data
        feature_columns = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        # Get labels data if present in testing input
        try:
            label_columns   = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
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

        # Prediction
        YpredCCF, _, _  = predictFromCCF(self.CCF, XTest)

        # Convert from ndarray from DataFrame
        YpredCCF_output = container.DataFrame(YpredCCF, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(YpredCCF_output.shape[1]):
            col_dict = dict(YpredCCF_output.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = self.label_name_columns[col]
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
            YpredCCF_output.metadata    = YpredCCF_output.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
        # Rename Columns to match label columns
        YpredCCF_output.columns = self.label_name_columns

        # Append to outputs
        outputs = outputs.append_columns(YpredCCF_output)

        return base.CallResult(outputs)


    def get_params(self) -> Params:
        return None

    def set_params(self, *, params: Params) -> None:
        return None
