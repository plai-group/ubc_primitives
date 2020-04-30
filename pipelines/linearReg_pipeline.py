import os
import sys
from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Common Primitives
from common_primitives import construct_predictions
from common_primitives.denormalize import DenormalizePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

# Testing Primitive
from primitives.vgg.vggnetcnn import VGG16CNN
from primitives.pca.pca import PrincipalComponentAnalysisPrimitive
from primitives.linearRegression.linear_regression import LinearRegressionPrimitive

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
    step_2.add_hyperparameter(name='include_top', argument_type=ArgumentType.VALUE, data=True)
    step_2.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: PCA
    step_3 = PrimitiveStep(primitive_description=PrincipalComponentAnalysisPrimitive.metadata.query())
    step_3.add_hyperparameter(name='max_components', argument_type=ArgumentType.VALUE, data=128)
    step_3.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # Step 4: Linear Regression
    step_4 = PrimitiveStep(primitive_description=LinearRegressionPrimitive.metadata.query())
    step_4.add_hyperparameter(name='learning_rate',    argument_type=ArgumentType.VALUE, data=0.001)
    step_4.add_hyperparameter(name='use_gradient_fit', argument_type=ArgumentType.VALUE, data=True)
    step_4.add_hyperparameter(name='minibatch_size',   argument_type=ArgumentType.VALUE, data=10)
    step_4.add_hyperparameter(name='num_iterations',   argument_type=ArgumentType.VALUE, data=1000)
    step_4.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    pipeline.add_step(step_4)

    # step 5: Construct Output
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_5.add_argument(name='inputs',    argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_5.add_output('produce')
    pipeline.add_step(step_5)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.5.produce')

    # print(pipeline.to_json())

    with open('./linearReg_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline for hand geometery dataset
    make_pipeline()

if __name__ == '__main__':
    main()
