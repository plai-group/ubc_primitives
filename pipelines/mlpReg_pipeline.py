import argparse

from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Common Primitives
from common_primitives import construct_predictions
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.dataframe_image_reader import DataFrameImageReaderPrimitive
from primitives.vgg.vggnetcnn import VGG16CNN

# Testing Primitive
from primitives.regMLP.mlpReg import MultilayerPerceptronRegressionPrimitive


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

    # Step 2: Feature Extraction Primitive
    step_2 = PrimitiveStep(primitive=VGG16CNN)
    step_2.add_hyperparameter(name='feature_extract_only', argument_type=ArgumentType.VALUE, data=True)
    step_2.add_hyperparameter(name='include_top', argument_type=ArgumentType.VALUE, data=False)
    step_2.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: Feature Extraction Primitive
    step_3 = PrimitiveStep(primitive=MultilayerPerceptronRegressionPrimitive)
    step_3.add_hyperparameter(name='input_dim',       argument_type=ArgumentType.VALUE, data=25088)
    step_3.add_hyperparameter(name='depth',           argument_type=ArgumentType.VALUE, data=1)
    step_3.add_hyperparameter(name='use_batch_norm',  argument_type=ArgumentType.VALUE, data=True)
    step_3.add_hyperparameter(name='activation_type', argument_type=ArgumentType.VALUE, data='leaky_relu')
    step_3.add_hyperparameter(name='dataset_type',    argument_type=ArgumentType.VALUE, data='dataset_1')
    step_3.add_hyperparameter(name='num_iterations',  argument_type=ArgumentType.VALUE, data=1000)
    step_3.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # step 4: Construct output
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_4.add_argument(name='inputs',    argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_4.add_output('produce')
    pipeline.add_step(step_4)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.4.produce')

    # print(pipeline.to_json())

    with open('./mlpReg_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    make_pipeline()


if __name__ == '__main__':
    # Generate pipeline for Hand geometry dataset
    main()
