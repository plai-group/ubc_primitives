import os
import sys
from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Common Primitives
from common_primitives import construct_predictions
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

# Testing Primitive
from primitives_ubc.logisticRegression.logistic_regression import LogisticRegressionPrimitive

def make_pipeline():
    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    # Step 1: DatasetToDataFrame
    step_0 = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Step 2: Logistic Regression
    step_1 = PrimitiveStep(primitive_description=LogisticRegressionPrimitive.metadata.query())
    step_1.add_hyperparameter(name='burnin', argument_type=ArgumentType.VALUE, data=10)
    step_1.add_hyperparameter(name='num_iterations', argument_type=ArgumentType.VALUE, data=5)
    step_1.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # step 3: Construct Output
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_2.add_argument(name='inputs',    argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.2.produce')

    # print(pipeline.to_json())

    with open('./logisticReg_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline for hand geometery dataset
    make_pipeline()

if __name__ == '__main__':
    main()
