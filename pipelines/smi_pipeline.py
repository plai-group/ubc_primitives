from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Common Primitive
from common_primitives import construct_predictions
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.remove_semantic_types import RemoveSemanticTypesPrimitive

import d3m.primitives.data_cleaning.imputer as Imputer
import d3m.primitives.classification.random_forest as RF

# Testing primitive
from primitives.smi.semantic_type import SemanticTypeInfer

def make_pipeline():
    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    # Step 0: DatasetToDataFrame
    step_0 = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Step 1: Call SMI primitive
    step_1 = PrimitiveStep(primitive=SemanticTypeInfer)
    step_1.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Step 2: Column Parser
    step_2 = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: Extract Attributes
    step_3 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
    pipeline.add_step(step_3)

    # Step 4: Extract Targets
    step_4 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
    pipeline.add_step(step_4)

    attributes = 'steps.3.produce'
    targets    = 'steps.4.produce'

    # Step 6: Imputer
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_5.add_output('produce')
    pipeline.add_step(step_5)

    # Step 7: SVC
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.decision_tree.SKlearn'))
    step_6.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER,  data_reference='steps.5.produce')
    step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    step_6.add_output('produce')
    pipeline.add_step(step_6)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.6.produce')

    # print(pipeline.to_json())

    with open('./semantic_type_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline for 38_sick_dataset
    make_pipeline()

if __name__ == '__main__':
    main()
