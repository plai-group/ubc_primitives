import os
import numpy as np
import pandas as pd
import unittest

from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.xgboost_regressor import XGBoostGBTreeRegressorPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

# Testing primitive
from primitives.cnn import ConvolutionalNeuralNetwork


class TestConvolutionalNeuralNetwork(unittest.TestCase):
    def test_1(self):
        """
        Feature extraction only and Testing on seed dataset from D3M datasets
        """
        print('\n')
        print('########################')
        print('#--------TEST-1--------#')
        print('########################')

        # Get volumes:
        all_weights = os.listdir('./static')
        all_weights = {w: os.path.join('./static', w) for w in all_weights}

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

        extractA_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extractA_hyperparams_class = extractA_hyperparams_class.defaults().replace(
                {
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/FileName',)
                }
        )
        extractA_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extractA_hyperparams_class)
        extractA = extractA_primitive.produce(inputs=dataframe.value)
        print(extractA.value)
        print('------------------------')

        extractP_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extractP_hyperparams = extractP_hyperparams_class.defaults().replace(
                {
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)
                }
        )
        extractP_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extractP_hyperparams)
        extractP = extractP_primitive.produce(inputs=dataframe.value)
        print(extractP.value)
        print('------------------------')

        # Call primitives
        hyperparams_class = ConvolutionalNeuralNetwork.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams_class = hyperparams_class.defaults().replace(
                {
                'feature_extract_only': False,
                'cnn_type': 'mobilenet',
                }
        )
        primitive = ConvolutionalNeuralNetwork(hyperparams=hyperparams_class, volumes=all_weights)
        primitive.set_training_data(inputs = dataframe.value, outputs = extractP.value)
        test_out  = primitive.fit(iterations=100)
        test_out  = primitive.produce(inputs=score.value)
        test_out  = test_out.value

        print(test_out)
        print('------------------------')
        for col in range(test_out.shape[1]):
            col_dict = dict(test_out.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)

        # Computer Error
        ground_truth = ((score.value['WRISTBREADTH']).to_numpy()).astype(np.float)
        predictions  = (test_out.iloc[:, -1]).to_numpy()

        print(ground_truth)
        print(predictions)
        print('------------------------')

        print('Mean squared error (lower better): ', (np.mean((predictions - ground_truth)**2)))
        print('------------------------')


    def test_2(self):
        """
        Training and Testing on seed dataset from D3M datasets
        """
        print('\n')
        print('########################')
        print('#--------TEST-2--------#')
        print('########################')

        # Get volumes:
        all_weights = os.listdir('./static')
        all_weights = {w: os.path.join('./static', w) for w in all_weights}

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
        hyperparams_class = ConvolutionalNeuralNetwork.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams_class = hyperparams_class.defaults().replace(
                {
                'cnn_type': 'googlenet',
                }
        )
        primitive = ConvolutionalNeuralNetwork(hyperparams=hyperparams_class.defaults(), volumes=all_weights)
        test_out  = primitive.produce(inputs=dataframe.value)

        print(test_out)
        print('------------------------')

        extractA_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extractA_hyperparams_class = extractA_hyperparams_class.defaults().replace(
                {
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',)
                }
        )
        extractA_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extractA_hyperparams_class)
        extractA = extractA_primitive.produce(inputs=test_out.value)
        print(extractA.value)
        print('------------------------')

        extractP_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extractP_hyperparams = extractP_hyperparams_class.defaults().replace(
                {
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)
                }
        )
        extractP_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extractP_hyperparams)
        extractP = extractP_primitive.produce(inputs=dataframe.value)
        extractP = extractP.value
        # Update Metadata from SuggestedTarget to TrueTarget
        for col in range((extractP).shape[1]):
            col_dict = dict(extractP.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = "WRISTBREADTH"
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/TrueTarget",)
            extractP.metadata           = extractP.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

        print(extractP)
        print('------------------------')

        # Call primitives
        score_out = primitive.produce(inputs=score.value)

        XGB_hyperparams_class = XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        XGB_primitive = XGBoostGBTreeRegressorPrimitive(hyperparams=XGB_hyperparams_class.defaults())
        XGB_primitive.set_training_data(inputs=test_out.value, outputs=extractP)
        XGB_primitive.fit()
        test_out_xgb = XGB_primitive.produce(inputs=score_out.value)
        test_out_xgb = test_out_xgb.value

        print('Predictions')
        print(test_out_xgb)
        print('------------------------')

        # Computer Error
        ground_truth = ((score.value['WRISTBREADTH']).to_numpy()).astype(np.float)
        predictions  = (test_out_xgb.iloc[:, -1]).to_numpy()

        print(ground_truth)
        print(predictions)
        print('------------------------')

        print('Mean squared error (lower better): ', (np.mean((predictions - ground_truth)**2)))
        print('------------------------')


if __name__ == '__main__':
    unittest.main()
