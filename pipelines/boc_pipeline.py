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

# Classification Primitive
import d3m.primitives.classification.random_forest as RF

# Testing Primitive
from primitives.boc.bag_of_characters import BagOfCharacters

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
    step_2 = PrimitiveStep(primitive=BagOfCharacters)
    step_2.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: Column Parser
    step_3 = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # Step 4: Extract Attributes
    step_4 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
    pipeline.add_step(step_4)

    # Step 5: Extract Targets
    step_5 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_5.add_output('produce')
    step_5.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
    pipeline.add_step(step_5)

    attributes = 'steps.4.produce'
    targets    = 'steps.5.produce'

    # Step 6: SVC
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.decision_tree.SKlearn'))
    step_6.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER,  data_reference='steps.4.produce')
    step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    step_6.add_output('produce')
    pipeline.add_step(step_6)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.6.produce')

    # print(pipeline.to_json())

    with open('./boc_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline for LL1_TXT_CLS_apple_products_sentiment dataset
    make_pipeline()

if __name__ == '__main__':
    main()
