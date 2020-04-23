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
from primitives.simpleCNAPS import SimpleCNAPSClassifierPrimitive


class TestSimpleCNAPSClassifierPrimitive(unittest.TestCase):
    # def test_0(self):
    #     """
    #     model Test
    #     """
    #     # Get volumes:
    #     all_weights = os.listdir('./static')
    #     all_weights = {w: os.path.join('./static', w) for w in all_weights}
    #
    #     print(all_weights)
    #
    #     print('CNAP Primitive....')
    #     cnaps_hyperparams_class = SimpleCNAPSClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     cnaps_hyperparams = cnaps_hyperparams_class.defaults()
    #     cnaps_primitive   = SimpleCNAPSClassifierPrimitive(hyperparams=cnaps_hyperparams, volumes=all_weights)

    # def test_1(self):
    #     """
    #     Dataset test
    #     """
    #     print('\n')
    #     print('########################')
    #     print('#--------TEST-1--------#')
    #     print('########################')
    #
    #     # Get volumes:
    #     all_weights = os.listdir('./static')
    #     all_weights = {w: os.path.join('./static', w) for w in all_weights}
    #
    #     # Loading dataset.
    #     path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/my_datasets/contributed_datasets/LWLL1_metadataset/TRAIN/dataset_TRAIN/datasetDoc.json'))
    #     dataset = Dataset.load(dataset_uri=path1)
    #
    #     # Step 0: Denormalize primitive
    #     denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
    #     denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
    #     denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
    #     denormalized_dataset  = denormalized_dataset.value
    #     print(denormalized_dataset)
    #     print('------------------------')
    #
    #     print('Loading Training Dataset....')
    #     # Step 1: Dataset to DataFrame
    #     dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
    #     dataframe = dataframe_primitive.produce(inputs=denormalized_dataset)
    #     dataframe = dataframe.value
    #     print(dataframe)
    #     print('------------------------')
    #
    #     print('CNAP Primitive....')
    #     cnaps_hyperparams_class = SimpleCNAPSClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     cnaps_hyperparams = cnaps_hyperparams_class.defaults()
    #     cnaps_primitive   = SimpleCNAPSClassifierPrimitive(hyperparams=cnaps_hyperparams)
    #     cnaps_primitive.set_training_data(inputs=dataframe, outputs=dataframe)
    #     cnaps_primitive.fit()
    #
    # def test_2(self):
    #     """
    #     Dataset test
    #     """
    #     print('\n')
    #     print('########################')
    #     print('#--------TEST-2--------#')
    #     print('########################')
    #
    #     # Get volumes:
    #     all_weights = os.listdir('./static')
    #     all_weights = {w: os.path.join('./static', w) for w in all_weights}
    #
    #     # Loading dataset.
    #     path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/my_datasets/contributed_datasets/LWLL1_metadataset/TEST/dataset_TEST/datasetDoc.json'))
    #     dataset = Dataset.load(dataset_uri=path1)
    #
    #     # Step 0: Denormalize primitive
    #     denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
    #     denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
    #     denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
    #     denormalized_dataset  = denormalized_dataset.value
    #     print(denormalized_dataset)
    #     print('------------------------')
    #
    #     print('Loading Training Dataset....')
    #     # Step 1: Dataset to DataFrame
    #     dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
    #     dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
    #     dataframe = dataframe_primitive.produce(inputs=denormalized_dataset)
    #     dataframe = dataframe.value
    #     print(dataframe)
    #     print('------------------------')
    #
    #     print('CNAP Primitive....')
    #     cnaps_hyperparams_class = SimpleCNAPSClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     cnaps_hyperparams = cnaps_hyperparams_class.defaults()
    #     cnaps_primitive   = SimpleCNAPSClassifierPrimitive(hyperparams=cnaps_hyperparams)
    #     cnaps_primitive.produce(inputs=dataframe)

    def test_3(self):
        """
        Dataset test
        """
        print('\n')
        print('########################')
        print('#--------TEST-3--------#')
        print('########################')

        # Get volumes:
        all_weights = os.listdir('./static')
        all_weights = {w: os.path.join('./static', w) for w in all_weights}

        # Loading dataset.
        path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc/cs/research/plai-scratch/tonyj/ubc_primitives/my_datasets/contributed_datasets/LWLL1_metadataset/SCORE/dataset_TEST/datasetDoc.json'))
        dataset = Dataset.load(dataset_uri=path1)

        # Step 0: Denormalize primitive
        denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
        denormalized_dataset  = denormalized_dataset.value
        print(denormalized_dataset)
        print('------------------------')

        print('Loading Training Dataset....')
        # Step 1: Dataset to DataFrame
        dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=denormalized_dataset)
        dataframe = dataframe.value
        print(dataframe)
        print('------------------------')

        print('CNAP Primitive....')
        cnaps_hyperparams_class = SimpleCNAPSClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        cnaps_hyperparams = cnaps_hyperparams_class.defaults()
        cnaps_primitive   = SimpleCNAPSClassifierPrimitive(hyperparams=cnaps_hyperparams, volumes=all_weights)
        cnaps_primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        cnaps_primitive.fit()
        outputs = cnaps_primitive.produce(inputs=dataframe)
        outputs = outputs.value

        print(outputs)


if __name__ == '__main__':
    unittest.main()
