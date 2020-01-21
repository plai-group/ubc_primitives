from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import Context, ArgumentType
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.column_parser import ColumnParserPrimitive

# Testing primitive
from primitives.smi.semantic_type import SemanticTypeInfer



def main():
    # DATASETS = 'LL1_net_nomination_seed'

    pipeline = meta_pipeline.Pipeline()
    pipeline.add_input(name='inputs')

    # Dataset To DataFrame
    step_0 = meta_pipeline.PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step_0.add_argument(name='inputs',
                        argument_type=ArgumentType.CONTAINER,
                        data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Parse Dataset
    step_1 = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step_1.add_hyperparameter(name='use_columns', argument_type=ArgumentType.VALUE, data=tuple(i for i in range(30)))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Call SMI primitive
    step_2 = PrimitiveStep(primitive_description=SemanticTypeInfer.metadata.query())
    step_2.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Final Output
    pipeline.add_output(name='results', data_reference='steps.2.produce')

    # Output to YAML
    print(pipeline_description.to_yaml())


if __name__ == '__main__':
    main()
