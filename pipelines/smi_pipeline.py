from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

# Testing primitive
from primitives.smi.semantic_type import SemanticTypeInfer

def make_pipeline():
    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    # Step 1: dataset_to_dataframe
    step_0 = PrimitiveStep(primitive=DatasetToDataFramePrimitive)
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Call SMI primitive
    step_1 = PrimitiveStep(primitive=SemanticTypeInfer)
    step_1.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Final Output
    pipeline.add_output(name='results', data_reference='steps.1.produce')

    # Output to JSON
    # print(pipeline.to_json())

    with open('./semantic_type_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline
    make_pipeline()

if __name__ == '__main__':
    main()
