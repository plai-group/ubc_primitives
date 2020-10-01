from d3m import container
from d3m.container import pandas # type: ignore
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams
from d3m.base import utils as base_utils

# Import config file
from primitives_ubc.config_files import config

# Import relevant libraries
import os
import math
import shutil
import string
import random
import typing
import logging
import importlib
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import OrderedDict

__all__ = ('BagOfCharacters',)

logger  = logging.getLogger(__name__)
Inputs  = container.DataFrame
Outputs = container.DataFrame


class ImportModules:
    """
    Import modules after calling the primitive as not to slow down d3m.index
    """
    _weight_files = []
    def __init__(self):
        self._initialized = False

    def _import_lib(self):
        if self._initialized:
            return
        global nltk, sy_stats

        nltk     = importlib.import_module('nltk')
        sy_stats = importlib.import_module('scipy.stats')
        self._initialized = True

class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters
    """
    n_samples = hyperparams.Constant(
        default=1000,
        description="Max number of samples/words to select",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class BagOfCharacters(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams], ImportModules):
    """
    A primitive for extract features describing the distribution of characters in a column.
    It computes the count of all 96 ASCII-printable characters (i.e., digits, letters,
    and punctuation characters, but not whitespace) within each value of a column.
    Then aggregate these counts with 10 statistical functions (i.e., any, all, mean,
    variance, min, max, median, sum, kurtosis, skewness), resulting in 960 features
    Citation: https://arxiv.org/pdf/1905.10688.pdf
    """

    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "5df70f68-b78f-4104-81ab-a935f4007316",
        "version": config.VERSION,
        "name": "Bag of Characters",
        "description": "A primitive for extract features describing the distribution of characters in a column",
        "python_path": "d3m.primitives.feature_extraction.boc.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.VECTORIZATION],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ["bag of characters", "NLP", "character features"],
        "installation": [config.INSTALLATION],
    })

    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Union[typing.Dict[str, str], None]=None):
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        self.hyperparams = hyperparams

        # Intialize LoadWeightsPrimitive
        ImportModules.__init__(self)

        # Import other needed modules
        ImportModules._import_lib(self)


    def _curate_data(self, inputs):
        """
        Process DataFrame
        """
        if inputs is None:
            raise exceptions.InvalidStateError("Missing data.")

        in_text_file=False;
        for col in range(inputs.shape[1]):
            col_dict = dict(inputs.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            if ('https://metadata.datadrivendiscovery.org/types/FileName' in col_dict['semantic_types']) and ('text/plain' in col_dict['media_types']):
                in_text_file = True

        if in_text_file:
            # Get all Nested media files
            text_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName') # [1]
            if len(text_columns) == 0:
                text_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute') # [1]
            base_paths    = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in text_columns] # Path + media
            txt_paths     = [[os.path.join(base_path, filename) for filename in inputs.iloc[:,col]] for base_path, col in zip(base_paths, text_columns)]
            # Extract the data from media files
            all_txts_ = []
            for path_list in txt_paths:
                interm_path = []
                for path in path_list:
                    open_file    = open(path, "r")
                    path_content = open_file.read().replace('\n', '')
                    open_file.close() # Close file to save Memory resource
                    interm_path.append(path_content)
                all_txts_.append(interm_path)
            # Final Input data
            all_txts_ = pd.DataFrame(all_txts_)
            all_txts  = all_txts_.T

            # Concatenate with text columns that aren't stored in nested files
            local_text_columns = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Text')
            local_text_columns = [col for col in local_text_columns if col not in text_columns]

            all_txts = pd.concat((all_txts, inputs[local_text_columns]), axis=1)

            return all_txts, text_columns

        else:
            # Get training data and labels data
            attribute_columns = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
            # # Get labels data if present in training input
            # try:
            #     label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            # except:
            #     label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            # # If no error but no label-columns found, force try SuggestedTarget
            # if len(label_columns) == 0 or label_columns == None:
            #     label_columns  = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            # # Remove columns if outputs present in inputs
            # if len(label_columns) >= 1:
            #     for lbl_c in label_columns:
            #         try:
            #             attribute_columns.remove(lbl_c)
            #         except ValueError:
            #             pass

            # Training Set
            attribute_columns = [int(ac) for ac in attribute_columns]
            attributes = inputs.iloc[:, attribute_columns]

            return attributes, attribute_columns


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Inputs: pandas DataFrame
        Returns: Output pandas DataFrame with 960 features.
        """
        attributes, attribute_columns = self._curate_data(inputs=inputs)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(attribute_columns)

        ### Build Features ###
        logging.info('Building Features in progress......')
        df_char = pd.DataFrame()
        counter = 0

        for name, raw_sample in attributes.iterrows():
            if counter % 1000 == 0:
                logging.info('Completion {}/{}'.format(counter, len(inputs)))

            n_values = len(raw_sample.values[0]) # Because inside a list

            if self.hyperparams['n_samples'] > n_values:
                n_samples = n_values
            else:
                n_samples = self.hyperparams['n_samples']

            # Sample n_samples from data column, and convert cell values to string values
            raw_sample = raw_sample.values[0] # Because inside a list
            raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

            # Extract Features
            extract_features = self._extract_bag_of_characters_features(raw_sample)
            df_char = df_char.append(extract_features, ignore_index=True)

            # Increment the progress counter
            counter += 1

        # Missing Values
        df_char.fillna(df_char.mean(), inplace=True)
        logging.info('Completion {}/{}'.format(len(inputs), len(inputs)))

        # Features
        feature_vectors = container.DataFrame(df_char, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(feature_vectors.shape[1]):
            col_dict = dict(feature_vectors.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = "feature_vector_" + str(col)
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
            feature_vectors.metadata    = feature_vectors.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

        # Add the features to the input labels with data removed
        outputs = outputs.append_columns(feature_vectors)

        return base.CallResult(outputs)


    def _extract_bag_of_characters_features(self, data:pd.DataFrame):
        """
        # Input Data: (pandas series) A single column.
        # Output: Ordered dictionary holding bag of character features
        """
        characters_to_check = [ '['+ c + ']' for c in string.printable if c not in ( '\n', '\\', '\v', '\r', '\t', '^')] + ['[\\\\]', '[\^]']

        f = OrderedDict()

        data_no_null = data.dropna()
        all_value_features = OrderedDict()

        for c in characters_to_check:
            all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)

        for value_feature_name, value_features in all_value_features.items():
            f['{}-agg-any'.format(value_feature_name)]      = any(value_features)
            f['{}-agg-all'.format(value_feature_name)]      = all(value_features)
            f['{}-agg-mean'.format(value_feature_name)]     = np.mean(value_features)
            f['{}-agg-var'.format(value_feature_name)]      = np.var(value_features)
            f['{}-agg-min'.format(value_feature_name)]      = np.min(value_features)
            f['{}-agg-max'.format(value_feature_name)]      = np.max(value_features)
            f['{}-agg-median'.format(value_feature_name)]   = np.median(value_features)
            f['{}-agg-sum'.format(value_feature_name)]      = np.sum(value_features)
            f['{}-agg-kurtosis'.format(value_feature_name)] = sy_stats.kurtosis(value_features)
            f['{}-agg-skewness'.format(value_feature_name)] = sy_stats.skew(value_features)

        return f
