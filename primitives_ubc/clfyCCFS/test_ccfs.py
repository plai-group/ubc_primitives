import sys
import os
import numpy as np
import pandas as pd
import unittest

from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.dataframe_to_ndarray import DataFrameToNDArrayPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

# sys.path.insert(0, os.path.abspath('/ubc_primitives/primitives/bow/'))
# from bag_of_words import BagOfWords

# Testing primitive
from ccfsClfy import CanonicalCorrelationForestsClassifierPrimitive

import warnings
warnings.filterwarnings('ignore')
# Ignore
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestCanonicalCorrelationForestsClassifierPrimitive(unittest.TestCase):
    # @ignore_warnings
    # def test_1(self):
    #     print('running test-1..............')
    #     import scipy.io
    #     from collections  import OrderedDict
    #     from primitives.clfyCCFS.src.generate_CCF import genCCF
    #     from primitives.clfyCCFS.src.predict_from_CCF import predictFromCCF
    #     from primitives.clfyCCFS.src.plotting.plot_surface import plotCCFDecisionSurface
    #     # Test for spiral dataset
    #     #-------------------------Use optionsClassCCF--------------------------#
    #     # Default HyperParams
    #     optionsClassCCF = {}
    #     optionsClassCCF['lambda']           = 'log'
    #     optionsClassCCF['splitCriterion']   = 'info'
    #     optionsClassCCF['minPointsLeaf']    = 1
    #     optionsClassCCF['bUseParallel']     = 1
    #     optionsClassCCF['bCalcTimingStats'] = 1
    #     optionsClassCCF['bSepPred']         = False
    #     optionsClassCCF['taskWeights']      = 'even'
    #     optionsClassCCF['bProjBoot']        = True
    #     optionsClassCCF['bBagTrees']        = True
    #     optionsClassCCF['projections']      = OrderedDict() # To ensure consistent order
    #     optionsClassCCF['projections']['CCA'] = True
    #     optionsClassCCF['treeRotation']       = 'none'
    #     optionsClassCCF['propTrain']          = 1
    #     optionsClassCCF['epsilonCCA']         = 1.0000e-04
    #     optionsClassCCF['mseErrorTolerance']  = 1.0000e-06
    #     optionsClassCCF['maxDepthSplit'] = 'stack'
    #     optionsClassCCF['XVariationTol'] = 1.0e-10
    #     optionsClassCCF['RotForM']  = 3
    #     optionsClassCCF['RotForpS'] = 0.7500
    #     optionsClassCCF['RotForpClassLeaveOut'] = 0.5000
    #     optionsClassCCF['minPointsForSplit']    = 2
    #     optionsClassCCF['dirIfEqual'] = 'first'
    #     optionsClassCCF['bContinueProjBootDegenerate'] = 1
    #     optionsClassCCF['multiTaskGainCombination'] = 'mean'
    #     optionsClassCCF['missingValuesMethod'] = 'random'
    #     optionsClassCCF['bUseOutputComponentsMSE'] = 0
    #     optionsClassCCF['bRCCA'] = 0
    #     optionsClassCCF['rccaLengthScale'] = 0.1000
    #     optionsClassCCF['rccaNFeatures'] = 50
    #     optionsClassCCF['rccaRegLambda'] = 1.0000e-03
    #     optionsClassCCF['rccaIncludeOriginal'] = 0
    #     optionsClassCCF['classNames'] = np.array([])
    #     optionsClassCCF['org_muY']    = np.array([])
    #     optionsClassCCF['org_stdY']   = np.array([])
    #     optionsClassCCF['mseTotal']   = np.array([])
    #     optionsClassCCF['task_ids']   = np.array([])
    #
    #     # print(optionsClassCCF)
    #     #-----------------------------------------------------------------------
    #     # Load data
    #     # To view the working of CCFS using a spiral dataset
    #     Tdata  = scipy.io.loadmat('/ccfs_ubc/datasets/seed_datasets_current/test_dataset/spiral.mat')
    #     XTrain = Tdata['XTrain']
    #     YTrain = Tdata['YTrain']
    #     XTest  = Tdata['XTest']
    #     YTest  = Tdata['YTest']
    #     print('Dataset Loaded!')
    #
    #     # Call CCF
    #     print('CCF.......')
    #     CCF = genCCF(XTrain, YTrain, nTrees=100, optionsFor=optionsClassCCF)
    #     YpredCCF, _, _ = predictFromCCF(CCF, XTest)
    #     print('CCF Test missclassification rate (lower better): ', (100*(1- np.mean(YTest==(YpredCCF), axis=0))),  '%')
    #
    #     #-----------------------------------------------------------------------
    #     # Plotting
    #     x1Lims = [np.round(np.min(XTrain[:, 0])-1), np.round(np.max(XTrain[:, 0])+2)]
    #     x2Lims = [np.round(np.min(XTrain[:, 1])-1), np.round(np.max(XTrain[:, 1])+2)]
    #
    #     plotCCFDecisionSurface("spiral_contour.svg", CCF, x1Lims, x2Lims, XTrain, X=XTest, Y=YTest)
    #     #-----------------------------------------------------------------------

    # @ignore_warnings
    # def test_2(self):
    #     print('\n')
    #     print('running test-2..............')
    #     # Loading training dataset.
    #     path1 = 'file://{uri}'.format(uri=os.path.abspath('/ccfs_ubc/datasets/seed_datasets_current/LL1_TXT_CLS_apple_products_sentiment/TRAIN/dataset_TRAIN/datasetDoc.json'))
    #     dataset = Dataset.load(dataset_uri=path1)
    #
    #     # Step 0: Denormalize primitive
    #     denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
    #     denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
    #     denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
    #
    #     print(denormalized_dataset.value)
    #     print('------------------------')
    #
    #     # Step 1: Dataset to DataFrame
    #     dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
    #     dataframe = dataframe_primitive.produce(inputs=denormalized_dataset.value)
    #
    #     print(dataframe.value)
    #     print('------------------------')
    #
    #     # Step 2: Profiler
    #     spp_hyperparams_class = SimpleProfilerPrimitive.metadata.get_hyperparams()
    #     spp_primitive = DatasetToDataFramePrimitive(hyperparams=spp_hyperparams_class.defaults())
    #     dataframe = spp_primitive.produce(inputs=dataframe.value)
    #
    #     # Feature Extraction
    #     bow_hyperparams_class = BagOfWords.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     bow_hyperparams_class = bow_hyperparams_class.defaults().replace(
    #             {
    #             'n_samples': 2000
    #             }
    #     )
    #     bow_primitive = BagOfWords(hyperparams=bow_hyperparams_class)
    #     bow_output = bow_primitive.produce(inputs=dataframe.value)
    #     bow_output = bow_output.value
    #     print(bow_output)
    #     print('------------------------')
    #
    #     # for col in range(tonp_output.value.shape[1]):
    #     #     col_dict = dict(tonp_output.value.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #     #     print('Meta-data - {}'.format(col), col_dict)
    #     #
    #
    #     ccfs_hyperparams_class = CanonicalCorrelationForestsClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     ccfs_hyperparams = ccfs_hyperparams_class.defaults().replace(
    #             {
    #             'nTrees': 100
    #             }
    #     )
    #     ccfs_primitive = CanonicalCorrelationForestsClassifierPrimitive(hyperparams=ccfs_hyperparams)
    #     ccfs_primitive.set_training_data(inputs=bow_output, outputs=dataframe.value)
    #     ccfs_primitive.fit()
    #
    #     #-----------------------------------------------------------------------
    #     # Loading Testing dataset.
    #     path2 = 'file://{uri}'.format(uri=os.path.abspath('/ccfs_ubc/datasets/seed_datasets_current/LL1_TXT_CLS_apple_products_sentiment/SCORE/dataset_SCORE/datasetDoc.json'))
    #     dataset2 = Dataset.load(dataset_uri=path2)
    #
    #     # Step 0: Denormalize primitive
    #     score_denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
    #     score_denormalize_primitive = DenormalizePrimitive(hyperparams=score_denormalize_hyperparams_class.defaults())
    #     score_denormalized_dataset  = score_denormalize_primitive.produce(inputs=dataset2)
    #
    #     print(denormalized_dataset.value)
    #     print('------------------------')
    #
    #     # Step 1: Dataset to DataFrame
    #     score_dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     score_dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=score_dataframe_hyperparams_class.defaults())
    #     score_dataframe = score_dataframe_primitive.produce(inputs=score_denormalized_dataset.value)
    #
    #     print(score_dataframe.value)
    #     print('------------------------')
    #
    #     # Feature Extraction
    #     bow_hyperparams_class = BagOfWords.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     bow_hyperparams_class = bow_hyperparams_class.defaults().replace(
    #             {
    #             'n_samples': 2000
    #             }
    #     )
    #     score_bow_primitive = BagOfWords(hyperparams=bow_hyperparams_class)
    #     score_bow_output = score_bow_primitive.produce(inputs=score_dataframe.value)
    #     score_bow_output = score_bow_output.value
    #     print(score_bow_output)
    #     print('------------------------')
    #
    #     score = ccfs_primitive.produce(inputs=score_bow_output)
    #     score = score.value
    #
    #     for col in range(score.shape[1]):
    #         col_dict = dict(score.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #         print('Meta-data - {}'.format(col), col_dict)
    #
    #     # Computer Error
    #     ground_truth = ((score_dataframe.value['sentiment']).to_numpy()).astype(np.float)
    #     predictions  = ((score.iloc[:, -1]).to_numpy()).astype(np.float)
    #     print('------------------------')
    #     print('Predictions')
    #     print(predictions)
    #     print('------------------------')
    #     print('Ground Truth')
    #     print(ground_truth)
    #     print('------------------------')
    #
    #     print('------------------------')
    #     print('CCF Test missclassification rate (lower better):  ',  (100*(1 - np.mean(ground_truth==predictions))))
    #     print('------------------------')


    # @ignore_warnings
    # def test_3(self):
    #     print('\n')
    #     print('running test-3..............')
    #     # Loading training dataset.
    #     path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/datasets/seed_datasets_current/38_sick_MIN_METADATA/TRAIN/dataset_TRAIN/datasetDoc.json'))
    #     dataset = Dataset.load(dataset_uri=path1)
    #
    #     # Step 1: Dataset to DataFrame
    #     dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
    #     dataframe = dataframe_primitive.produce(inputs=dataset)
    #
    #     # print(dataframe.value)
    #     # print('------------------------')
    #     #
    #     # for col in range(dataframe.value.shape[1]):
    #     #     col_dict = dict(dataframe.value.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #     #     print('Meta-data - {}'.format(col), col_dict)
    #
    #     # Step 2: Profiler
    #     spp_hyperparams_class = SimpleProfilerPrimitive.metadata.get_hyperparams()
    #     spp_primitive = SimpleProfilerPrimitive(hyperparams=spp_hyperparams_class.defaults())
    #     spp_primitive.set_training_data(inputs=dataframe.value)
    #     spp_primitive.fit()
    #     dataframe = spp_primitive.produce(inputs=dataframe.value)
    #     dataframe = dataframe.value
    #
    #     # Update target column
    #     target_col = 30
    #     col_dict = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS, target_col)))
    #     col_dict["semantic_types"]  = ("https://metadata.datadrivendiscovery.org/types/CategoricalData", "https://metadata.datadrivendiscovery.org/types/TrueTarget",)
    #     dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, target_col), col_dict)
    #
    #     for col in range(dataframe.shape[1]):
    #         col_dict = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #         print('Meta-data - {}'.format(col), col_dict)
    #     print('------------------------')
    #
    #     # Step 3: Extract Attributes
    #     ecstp_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
    #     ecstp_hyperparams = ecstp_hyperparams_class.defaults().replace(
    #             {
    #             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
    #             }
    #     )
    #     ecstp_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=ecstp_hyperparams)
    #     inputs = ecstp_primitive.produce(inputs=dataframe)
    #
    #     print('--------INPUTS----------')
    #     print(inputs.value)
    #     print('------------------------')
    #
    #     for col in range(inputs.value.shape[1]):
    #         col_dict = dict(inputs.value.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #         print('Meta-data - {}'.format(col), col_dict)
    #
    #     # Step 4: Extract Targets
    #     ecstp_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
    #     ecstp_hyperparams = ecstp_hyperparams_class.defaults().replace(
    #             {
    #             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
    #             }
    #     )
    #     ecstp_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=ecstp_hyperparams)
    #     outputs = ecstp_primitive.produce(inputs=dataframe)
    #
    #     print('--------OUTPUTS---------')
    #     print(outputs.value)
    #     print('------------------------')
    #
    #     # Step 5: CCFs
    #     ccfs_hyperparams_class = CanonicalCorrelationForestsClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     ccfs_hyperparams = ccfs_hyperparams_class.defaults().replace(
    #             {
    #             'nTrees': 100
    #             }
    #     )
    #     ccfs_primitive = CanonicalCorrelationForestsClassifierPrimitive(hyperparams=ccfs_hyperparams)
    #     ccfs_primitive.set_training_data(inputs=inputs.value, outputs=outputs.value)
    #     ccfs_primitive.fit()
    #
    #
    #     #-----------------------------------------------------------------------
    #     # Loading Testing dataset.
    #     path2 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/datasets/seed_datasets_current/38_sick_MIN_METADATA/SCORE/dataset_SCORE/datasetDoc.json'))
    #     dataset2 = Dataset.load(dataset_uri=path2)
    #
    #     # Step 1: Dataset to DataFrame
    #     score_dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     score_dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=score_dataframe_hyperparams_class.defaults())
    #     score_dataframe = score_dataframe_primitive.produce(inputs=dataset2)
    #
    #     print(score_dataframe.value)
    #     print('------------------------')
    #
    #     # Step 2: Profiler
    #     spp_hyperparams_class = SimpleProfilerPrimitive.metadata.get_hyperparams()
    #     spp_primitive = SimpleProfilerPrimitive(hyperparams=spp_hyperparams_class.defaults())
    #     spp_primitive.set_training_data(inputs=score_dataframe.value)
    #     spp_primitive.fit()
    #     dataframe = spp_primitive.produce(inputs=score_dataframe.value)
    #     dataframe = dataframe.value
    #
    #     # Update target column
    #     target_col = 30
    #     col_dict = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS, target_col)))
    #     col_dict["semantic_types"]  = ("https://metadata.datadrivendiscovery.org/types/CategoricalData", "https://metadata.datadrivendiscovery.org/types/TrueTarget",)
    #     dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, target_col), col_dict)
    #
    #     for col in range(dataframe.shape[1]):
    #         col_dict = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #         print('Meta-data - {}'.format(col), col_dict)
    #     print('------------------------')
    #
    #     # Step 3: Extract Attributes
    #     ecstp_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
    #     ecstp_hyperparams = ecstp_hyperparams_class.defaults().replace(
    #             {
    #             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
    #             }
    #     )
    #     ecstp_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=ecstp_hyperparams)
    #     inputs = ecstp_primitive.produce(inputs=dataframe)
    #
    #     print('--------INPUTS----------')
    #     print(inputs.value)
    #     print('------------------------')
    #
    #     for col in range(inputs.value.shape[1]):
    #         col_dict = dict(inputs.value.metadata.query((metadata_base.ALL_ELEMENTS, col)))
    #         print('Meta-data - {}'.format(col), col_dict)
    #
    #     # Step 4: Extract Targets
    #     ecstp_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
    #     ecstp_hyperparams = ecstp_hyperparams_class.defaults().replace(
    #             {
    #             'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
    #             }
    #     )
    #     ecstp_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=ecstp_hyperparams)
    #     outputs = ecstp_primitive.produce(inputs=dataframe)
    #
    #     print('--------OUTPUTS---------')
    #     print(outputs.value)
    #     print('------------------------')
    #
    #     # Step 5: CCFs
    #     score = ccfs_primitive.produce(inputs=inputs.value)
    #     score = score.value
    #
    #     print('------CCFS-OUTPUTS------')
    #     print(score)
    #     print('------------------------')
    #
    #
    #     # Computer Error
    #     ground_truth = ((score_dataframe.value['Class']).to_numpy())
    #     predictions  = ((score.iloc[:, -1]).to_numpy())
    #     # print('------------------------')
    #     # print('Predictions')
    #     # print(predictions)
    #     # print('------------------------')
    #     # print('Ground Truth')
    #     # print(ground_truth)
    #     # print('------------------------')
    #
    #     print('------------------------')
    #     print('CCF Test classification rate (Higher better):  ',  (100*(np.mean(ground_truth==predictions))))
    #     print('------------------------')


if __name__ == '__main__':
    unittest.main()
