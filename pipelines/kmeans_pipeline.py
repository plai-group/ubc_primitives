from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Common Primitives
from common_primitives import construct_predictions
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

# Testing Primitive
from primitives_ubc.bow.bag_of_words import BagOfWords
from primitives_ubc.kmeans.kmeansClustering import KMeansClusteringPrimitive


def make_pipeline():
    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    # Step 0: Denormalize
    step_0 = PrimitiveStep(primitive_description=DenormalizePrimitive.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Step 1: DatasetToDataFrame
    step_1 = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Step 2: Profiler
    step_2 = PrimitiveStep(primitive_description=SimpleProfilerPrimitive.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: Feature Extraction Primitive
    step_3 = PrimitiveStep(primitive=BagOfWords)
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # Step 4: Clustering Primitive
    step_4 = PrimitiveStep(primitive=KMeansClusteringPrimitive)
    step_4.add_hyperparameter(name='n_clusters', argument_type=ArgumentType.VALUE, data=4)
    step_4.add_hyperparameter(name='n_init',     argument_type=ArgumentType.VALUE, data=10)
    step_4.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    pipeline.add_step(step_4)

    # step 5: Construct output
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_5.add_argument(name='inputs',    argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_5.add_output('produce')
    pipeline.add_step(step_5)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.5.produce')

    # print(pipeline.to_json())

    with open('./kmeans_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')


def main():
    make_pipeline()


if __name__ == '__main__':
    # Generate pipeline for LL1_TXT_CLS_apple_products_sentiment dataset
    main()
