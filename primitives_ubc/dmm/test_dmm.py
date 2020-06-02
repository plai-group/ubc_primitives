import os
import numpy as np
import pandas as pd
import unittest

from d3m.metrics import Metric
from d3m import container, exceptions, utils
from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

# Testing primitive
from primitives_ubc.dmm.deep_markov_model import DeepMarkovModelPrimitive


class TestDeepMarkovModelPrimitive(unittest.TestCase):
    def test_0(self):
        """
        model Test
        """
        print('Deep Markov Model Primitive....')
        dmm_hyperparams_class = DeepMarkovModelPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        dmm_hyperparams = dmm_hyperparams_class.defaults()
        dmm_primitive   = DeepMarkovModelPrimitive(hyperparams=dmm_hyperparams)

    def test_1(self):
        """
        Dataset test
        """
        print('\n')
        print('########################')
        print('#--------TEST-1--------#')
        print('########################')


        # Loading dataset.
        path1 = 'file://{uri}'.format(uri=os.path.abspath('/ubc_primitives/datasets/seed_datasets_current/LL1_736_stock_market/SCORE/dataset_SCORE/datasetDoc.json'))
        dataset = Dataset.load(dataset_uri=path1)

        # # Step 0: Denormalize primitive
        # denormalize_hyperparams_class = DenormalizePrimitive.metadata.get_hyperparams()
        # denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        # denormalized_dataset  = denormalize_primitive.produce(inputs=dataset)
        # denormalized_dataset  = denormalized_dataset.value
        # print(denormalized_dataset)
        # print('------------------------')

        print('Loading Training Dataset....')
        # Step 0: Dataset to DataFrame
        dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset)
        dataframe = dataframe.value
        print(dataframe)

        for col in range(dataframe.shape[1]):
            col_dict = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)
        print('------------------------')


        # Step 1: Profiler
        print('Profiler')
        profiler_hyperparams_class = SimpleProfilerPrimitive.metadata.get_hyperparams()
        profiler_primitive = SimpleProfilerPrimitive(hyperparams=profiler_hyperparams_class.defaults())
        profiler_primitive.set_training_data(inputs=dataframe)
        profiler_primitive.fit()
        profiler_dataframe = profiler_primitive.produce(inputs=dataframe)
        profiler_dataframe = profiler_dataframe.value
        print(profiler_dataframe)

        for col in range(profiler_dataframe.shape[1]):
            col_dict = dict(profiler_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)
        print('------------------------')


        # Step 2: Column parser
        print('Column parser')
        parser_hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        parser_hyperparams = parser_hyperparams_class.defaults().replace(
                {'parse_semantic_types': ["http://schema.org/Boolean",
                                          "http://schema.org/Integer",
                                          "http://schema.org/Float",
                                          "https://metadata.datadrivendiscovery.org/types/FloatVector",
                                          "http://schema.org/DateTime"]
                }
        )
        parser_primitive = ColumnParserPrimitive(hyperparams=parser_hyperparams)
        parser_dataframe = parser_primitive.produce(inputs=profiler_dataframe)
        parser_dataframe = parser_dataframe.value
        print(parser_dataframe)
        print('------------------------')

        for col in range(parser_dataframe.shape[1]):
            col_dict = dict(parser_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            print('Meta-data - {}'.format(col), col_dict)


        # Step 4: Extract dataframe
        print('Extract dataframe')
        extract_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
        extract_hyperparams = extract_hyperparams_class.defaults().replace(
                {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']
                }
        )
        extract_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extract_hyperparams)
        extract_dataframe = extract_primitive.produce(inputs=parser_dataframe)
        extract_dataframe = extract_dataframe.value
        print(extract_dataframe)
        print('------------------------')

        # Step 5: Extract target
        print('Extract target')
        extract_hyperparams_class = ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
        extract_hyperparams = extract_hyperparams_class.defaults().replace(
                {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                }
        )
        extract_primitive = ExtractColumnsBySemanticTypesPrimitive(hyperparams=extract_hyperparams)
        extract_targets = extract_primitive.produce(inputs=parser_dataframe)
        extract_targets = extract_targets.value
        print(extract_targets)
        print('------------------------')


        print('DMM Primitive....')
        dmm_hyperparams_class = DeepMarkovModelPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        dmm_hyperparams = dmm_hyperparams_class.defaults()
        dmm_primitive   = DeepMarkovModelPrimitive(hyperparams=dmm_hyperparams)
        dmm_primitive.set_training_data(inputs=extract_dataframe, outputs=extract_targets)
        print(dmm_primitive._training_inputs)
        dmm_primitive.fit()


if __name__ == '__main__':
    unittest.main()
