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

sys.path.insert(0, os.path.abspath('/ubc_primitives/primitives_ubc/bow/'))
from bag_of_words import BagOfWords

# Testing Primitive
from primitives_ubc.clfyCCFS.ccfsClfy import CanonicalCorrelationForestsClassifierPrimitive

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
    step_2 = PrimitiveStep(primitive_description=BagOfWords.metadata.query())
    step_2.add_hyperparameter(name='n_samples', argument_type=ArgumentType.VALUE, data=2000)
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: CCFs
    step_3 = PrimitiveStep(primitive_description=CanonicalCorrelationForestsClassifierPrimitive.metadata.query())
    step_3.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # step 4: Construct Output
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_4.add_argument(name='inputs',    argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_4.add_output('produce')
    pipeline.add_step(step_4)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.4.produce')

    # print(pipeline.to_json())

    with open('./clfyCCFS_pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

    print('Generated pipeline!')

def main():
    # Generate pipeline for LL1_TXT_CLS_apple_products_sentiment dataset
    make_pipeline()

if __name__ == '__main__':
    main()
