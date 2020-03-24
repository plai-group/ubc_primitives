import sys
import os
import numpy as np
import pandas as pd
import unittest

from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

sys.path.insert(0, os.path.abspath('/ubc_primitives/primitives/resnet/'))
from resnetcnn import ResNetCNN

# Testing primitive
from ccfsReg import CanonicalCorrelationForestsRegressionPrimitive

import warnings
warnings.filterwarnings('ignore')
# Ignore
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestCanonicalCorrelationForestsRegressionPrimitive(unittest.TestCase):
    @ignore_warnings
    def test_1(self):
        import scipy.io
        from collections  import OrderedDict
        from primitives.regCCFS.src.generate_CCF import genCCF
        from primitives.regCCFS.src.predict_from_CCF import predictFromCCF
        from primitives.regCCFS.src.plotting.plot_surface import plotCCFDecisionSurface
        # Test for spiral dataset
        #-------------------------Use optionsClassCCF--------------------------#
        # Default HyperParams
        optionsClassCCF = {}
        optionsClassCCF = {}
        optionsClassCCF['lambda']           = 'log'
        optionsClassCCF['splitCriterion']   = 'mse'
        optionsClassCCF['minPointsLeaf']    = 3
        optionsClassCCF['bUseParallel']     = 1
        optionsClassCCF['bCalcTimingStats'] = 1
        optionsClassCCF['bSepPred']         = False
        optionsClassCCF['taskWeights']      = 'even'
        optionsClassCCF['bProjBoot']        = True
        optionsClassCCF['bBagTrees']        = True
        optionsClassCCF['projections']      = OrderedDict() # To ensure consistent order
        optionsClassCCF['projections']['CCA'] = True
        optionsClassCCF['treeRotation']       = 'none'
        optionsClassCCF['propTrain']          = 1
        optionsClassCCF['epsilonCCA']         = 1.0000e-04
        optionsClassCCF['mseErrorTolerance']  = 1.0000e-06
        optionsClassCCF['maxDepthSplit'] = 'stack'
        optionsClassCCF['XVariationTol'] = 1.0e-10
        optionsClassCCF['RotForM']  = 3
        optionsClassCCF['RotForpS'] = 0.7500
        optionsClassCCF['RotForpClassLeaveOut'] = 0.5000
        optionsClassCCF['minPointsForSplit']    = 6
        optionsClassCCF['dirIfEqual'] = 'first'
        optionsClassCCF['bContinueProjBootDegenerate'] = True
        optionsClassCCF['multiTaskGainCombination'] = 'mean'
        optionsClassCCF['missingValuesMethod'] = 'random'
        optionsClassCCF['bUseOutputComponentsMSE'] = False
        optionsClassCCF['bRCCA'] = False
        optionsClassCCF['rccaLengthScale'] = 0.1000
        optionsClassCCF['rccaNFeatures'] = 50
        optionsClassCCF['rccaRegLambda'] = 1.0000e-03
        optionsClassCCF['rccaIncludeOriginal'] = 0
        optionsClassCCF['classNames'] = np.array([])
        optionsClassCCF['org_muY']    = np.array([])
        optionsClassCCF['org_stdY']   = np.array([])
        optionsClassCCF['mseTotal']   = np.array([])
        optionsClassCCF['task_ids']   = np.array([])

        # print(optionsClassCCF)
        #-----------------------------------------------------------------------
        # Load data
        Tdata  = scipy.io.loadmat('/ccfs_ubc/datasets/seed_datasets_current/test_dataset/camel6.mat')
        XTrain = Tdata['XTrain']
        YTrain = Tdata['YTrain']
        XTest  = Tdata['XTest']
        YTest  = Tdata['YTest']
        print(XTrain.shape)
        print(YTrain.shape)
        print('Dataset Loaded!')

        # Call CCF
        print('CCF.......')
        CCF = genCCF(XTrain, YTrain, nTrees=200, bReg=True, optionsFor=optionsClassCCF)
        YpredCCF, _, _ = predictFromCCF(CCF, XTest)
        print('CCF Mean squared error (lower better): ', (np.mean((YpredCCF - YTest)**2)))

        #-----------------------------------------------------------------------
        # Plotting
        x1Lims = [-1.15, 1.15]
        x2Lims = [-1.75, 1.75]

        plotCCFDecisionSurface("spiral_contour.svg", CCF, x1Lims, x2Lims, XTrain, X=XTest, Y=YTest, plot_X=False)
        # -----------------------------------------------------------------------

    @ignore_warnings
    def test_2(self):
        print('\n')
        # Get volumes:
        all_weights = os.listdir('/ubc_primitives/primitives/resnet/static')
        all_weights = {w: os.path.join('/ubc_primitives/primitives/resnet/static', w) for w in all_weights}
        # Loading dataset.
        path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/datasets/seed_datasets_current/22_handgeometry/TRAIN/dataset_TRAIN/datasetDoc.json'))
        dataset = Dataset.load(dataset_uri=path1)
        # Get dataset paths
        path2 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/datasets/seed_datasets_current/22_handgeometry/SCORE/dataset_TEST/datasetDoc.json'))
        score_dataset = Dataset.load(dataset_uri=path2)

        # Step 0: Denormalize primitive
        denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
        print(denormalized_dataset.value)
        print('------------------------')

        print('Loading Training Dataset....')
        # Step 1: Dataset to DataFrame
        dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=denormalized_dataset.value)
        print(dataframe.value)
        print('------------------------')

        print('Loading Testing Dataset....')
        # Step 0: Denormalize primitive
        score_denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        score_denormalize_primitive = DenormalizePrimitive(hyperparams=score_denormalize_hyperparams_class.defaults())
        score_denormalized_dataset  = score_denormalize_primitive.produce(inputs=score_dataset)
        print(score_denormalized_dataset.value)
        print('------------------------')

        score_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        score_primitive = DatasetToDataFramePrimitive(hyperparams=score_hyperparams_class.defaults())
        score = score_primitive.produce(inputs=score_denormalized_dataset.value)
        print(score.value)
        print('------------------------')

        # Call primitives
        hyperparams_class = ResNetCNN.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams_class = hyperparams_class.defaults().replace(
                {
                'include_top': True,
                }
        )
        feature_primitive = ResNetCNN(hyperparams=hyperparams_class, volumes=all_weights)
        train_feature_out = feature_primitive.produce(inputs=dataframe.value)
        test_feature_out  = feature_primitive.produce(inputs=score.value)
        train_feature_out = train_feature_out.value
        test_feature_out  = test_feature_out.value

        print('-----------------------------')
        print('Completed Feature extraction!')
        print('-----------------------------')

        print(test_feature_out)

        ccfs_hyperparams_class = CanonicalCorrelationForestsRegressionPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ccfs_hyperparams = ccfs_hyperparams_class.defaults().replace(
                {
                'nTrees': 200,
                }
        )
        ccfs_primitive = CanonicalCorrelationForestsRegressionPrimitive(hyperparams=ccfs_hyperparams)
        ccfs_primitive.set_training_data(inputs=train_feature_out, outputs=dataframe.value)
        ccfs_primitive.fit()
        score_out = ccfs_primitive.produce(inputs=test_feature_out)
        score_out = score_out.value
        print('------------------------')
        print(score_out)
        print('------------------------')

        for col in range(score_out.shape[1]):
            col_dict = dict(score_out.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)

        # Computer Error
        ground_truth   = ((score.value['WRISTBREADTH']).to_numpy()).astype(np.float)
        predictions    = ((score_out.iloc[:, -1]).to_numpy()).astype(np.float)
        print('------------------------')
        print('Predictions')
        print(predictions)
        print('------------------------')
        print('Ground Truth')
        print(ground_truth)
        print('------------------------')

        print('------------------------')
        print('CCF Mean squared error (lower better): ', (np.mean((ground_truth - predictions)**2)))
        print('------------------------')

if __name__ == '__main__':
    unittest.main()
