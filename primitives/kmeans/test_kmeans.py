import sys
import os
import numpy as np
import pandas as pd
import unittest

from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe   import DatasetToDataFramePrimitive
from common_primitives. ndarray_to_dataframe  import NDArrayToDataFramePrimitive
from common_primitives.dataframe_image_reader import DataFrameImageReaderPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive

# Testing primitive
from primitives.bow.bag_of_words import BagOfWords
from kmeansClustering import KMeansClusteringPrimitive

import warnings
warnings.filterwarnings('ignore')
# Ignore
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestKMeansClusteringPrimitive(unittest.TestCase):
    def test_1(self):
        print('running test-1..............')
        # Checking network with some configuration
        kmeans_hyperparams_class = KMeansClusteringPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        kmeans_hyperparams = kmeans_hyperparams_class.defaults().replace(
                {
                'n_clusters': 10,
                }
        )
        kmeans_primitive = KMeansClusteringPrimitive(hyperparams=kmeans_hyperparams)
        print(kmeans_primitive._kmeans)
        print('------------------------')


    def test_1(self):
        print('\n')
        print('running test-2..............')
        # Loading training dataset.
        base_path = "/ubc_primitives/datasets/seed_datasets_current/LL1_TXT_CLS_apple_products_sentiment"
        dataset_doc_path = os.path.join(base_path,\
                                        'TRAIN/dataset_TRAIN',\
                                        'datasetDoc.json')
        dataset = Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # Step 0: Denormalize primitive
        denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)

        print(denormalized_dataset.value)
        print('------------------------')

        # Step 1: Dataset to DataFrame
        dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=denormalized_dataset.value)

        print(dataframe.value)
        print('------------------------')

        # Step 2: DataFrame to features
        bow_hyperparams_class = BagOfWords.metadata.get_hyperparams()
        bow_primitive = BagOfWords(hyperparams=bow_hyperparams_class.defaults())
        bow_primitive_out = bow_primitive.produce(inputs=dataframe.value)

        # Step 3: Dataset to DataFrame
        kmeans_hyperparams_class = KMeansClusteringPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        kmeans_hyperparams = kmeans_hyperparams_class.defaults().replace(
                {
                'n_clusters': 4,
                'n_init': 10,
                'max_iter': 1000
                }
        )
        kmeans_primitive = KMeansClusteringPrimitive(hyperparams=kmeans_hyperparams)
        kmeans_primitive.set_training_data(inputs=bow_primitive_out.value)
        kmeans_primitive.fit()

        #-----------------------------------------------------------------------
        # Loading Testing dataset.
        dataset_doc_path2 = os.path.join(base_path,\
                                         'SCORE/dataset_SCORE',\
                                         'datasetDoc.json')
        dataset2 = Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path2))

        # Step 0: Denormalize primitive
        score_denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        score_denormalize_primitive = DenormalizePrimitive(hyperparams=score_denormalize_hyperparams_class.defaults())
        score_denormalized_dataset  = score_denormalize_primitive.produce(inputs=dataset2)

        print(denormalized_dataset.value)
        print('------------------------')

        # Step 1: Dataset to DataFrame
        score_dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        score_dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=score_dataframe_hyperparams_class.defaults())
        score_dataframe = score_dataframe_primitive.produce(inputs=score_denormalized_dataset.value)

        print(score_dataframe.value)
        print('------------------------')

        # Step 2: Read images to DataFrame
        score_bow_dataframe = bow_primitive.produce(inputs=score_dataframe.value)

        print(score_bow_dataframe.value)
        print('------------------------')

        score = kmeans_primitive.produce(inputs=score_bow_dataframe.value)
        score = score.value

        print(score)
        print('------------------------')

        for col in range(score.shape[1]):
            col_dict = dict(score.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)

        # Computer Error
        ground_truth = ((score_dataframe.value['sentiment']).to_numpy()).astype(np.float)
        predictions  = ((score.iloc[:, -1]).to_numpy()).astype(np.float)
        print('------------------------')
        print('Predictions')
        print(predictions)
        print('------------------------')
        print('Ground Truth')
        print(ground_truth)
        print('------------------------')

        print('------------------------')
        print('MLP Test missclassification rate (lower better):  ',  (100*(1 - np.mean(ground_truth==predictions))))
        print('------------------------')

        # # Step 1: Dataset to DataFrame
        # cpp_dataframe_hyperparams_class = ConstructPredictionsPrimitive.metadata.get_hyperparams()
        # cpp_dataframe_primitive = ConstructPredictionsPrimitive(hyperparams=cpp_dataframe_hyperparams_class.defaults())
        # cpp_dataframe = cpp_dataframe_primitive.produce(inputs=score, reference=score_dataframe.value)
        #
        # print(cpp_dataframe.value)
        # print('------------------------')


if __name__ == '__main__':
    unittest.main()
