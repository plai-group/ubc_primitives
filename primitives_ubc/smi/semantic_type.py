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
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

__all__ = ('SemanticTypeInfer',)

logger  = logging.getLogger(__name__)
Inputs  = container.DataFrame
Outputs = container.DataFrame

class LoadWeightsPrimitive:
    """
    Loads primitives
    """
    _weight_files = []
    def __init__(self):
        self._initialized = False

    def _import_lib(self):
        if self._initialized:
            return
        # Import modules after calling the primitive,
        # as not to slow down d3m.index
        global tf, nltk, doc2vec, sy_stats

        tf       = importlib.import_module('tensorflow')
        nltk     = importlib.import_module('nltk')
        doc2vec  = importlib.import_module('gensim.models.doc2vec')
        sy_stats = importlib.import_module('scipy.stats')
        self._initialized = True

    @staticmethod
    def _get_weights_data_dir():
        """
        Return cache directory
        """
        datadir = '/static'

        if not os.access(datadir, os.W_OK):
            datadir = '.'

        return datadir

    @staticmethod
    def _get_weight_installation(weight_files: typing.List['WeightFile']):
        """
        Return D3M file installation entries
        """
        return [{'type': 'FILE',\
                 'key': weight_file.name,\
                 'file_uri': weight_file.uri,\
                 'file_digest': weight_file.digest} for weight_file in weight_files]

    @staticmethod
    def _get_weight_installation_tar(weight_files: typing.List['WeightFile']):
        """
        Return D3M file installation entries
        """
        return [{'type': 'TGZ',\
                 'key': weight_file.name,\
                 'file_uri': weight_file.uri,\
                 'file_digest': weight_file.digest} for weight_file in weight_files]

    @staticmethod
    def _find_weights_dir(key_filename, volumes):
        if key_filename in volumes:
            _weight_file_path = volumes[key_filename]
        else:
            static_dir = os.getenv('D3MSTATICDIR', '/static')
            _weight_file_path = os.path.join(static_dir, key_filename)

        if os.path.isfile(_weight_file_path):
            return _weight_file_path
        else:
            raise ValueError("Can't get weights file from the volume by key: {} or in the static folder: {}".format(key_filename, _weight_file_path))

        return _weight_file_path


class WeightFile(typing.NamedTuple):
    """
    Meta-data-->Installation: configs
    """
    name: str
    uri: str
    digest: str
    data_dir: str = LoadWeightsPrimitive._get_weights_data_dir()



class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters
    """
    use_row_iter = hyperparams.UniformBool(
        default=False,
        description="Whether or not to use row iteration inplace of column interation on dataframe",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be cast to the type, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )



class SemanticTypeInfer(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams], LoadWeightsPrimitive):
    """
    A primitive for detecting the semantic type of inputed column data.
    --> Currently Supported: 78 Semantic Types
     --------------------------------------------------------------------------
     |Address        | Code        | Education   | Notes        | Requirement |
     |Affiliate      | Collection  | Elevation   | Operator     | Result      |
     |Age            | Company     | File size   | Organisation | Service     |
     |Affiliation    | Command     | Family      | Order        | Sales       |
     |Album          | Component   | Format      | Origin       | Sex         |
     |Area           | Continent   | Gender      | Owner        | Species     |
     |Artist         | Country     | Genre       | Person       | State       |
     |Birth date     | County      | Grades      | Plays        | Status      |
     |Birth place    | Creator     | Industry    | Position     | Symbol      |
     |Brand          | Credit      | ISBN        | Product      | Team        |
     |Capacity       | Currency    | Jockey      | Publisher    | Team name   |
     |Category       | Day         | Language    | Range        | Type        |
     |City           | Depth       | Location    | Rank         | Weight      |
     |Class          | Description | Manufacturer| Ranking      | Year        |
     |Classification | Director    | Name        | Region       |             |
     |Club           | Duration    | Nationality | Religion     |             |
     --------------------------------------------------------------------------
    """
    ### Get Static files ###
    # All files directly downloaded from DropBox
    _weight_files_1 = [
        WeightFile('sherlock_weights.h5',
                   ('https://dl.dropboxusercontent.com/s/8g14nif72mp44o7/sherlock_weights.h5?dl=1'),
                   '4b121359def9f155c4e80728c9320a51b46c56b98c0e9949d3406ff6ba56dc14'),
        WeightFile('sherlock_model.json',
                   ('https://dl.dropboxusercontent.com/s/2bb9n3g1b982r04/sherlock_model.json?dl=1'),
                   'a12efdb386256a27f234eb475550cbb3ad4820bd5a5a085f6da4cdd36797897f'),
        WeightFile('classes_sherlock.npy',
                   ('https://dl.dropboxusercontent.com/s/k7mjisbfmffw4l4/classes_sherlock.npy?dl=1'),
                   '0bb18ba9dd97e124c8956f0abb1e8ff3a5aeabe619a3c38852d85ea0ec876c4a'),
        WeightFile('glove.6B.50d.txt',
                   ('https://dl.dropboxusercontent.com/s/8x197jze94d82qu/glove.6B.50d.txt?dl=1'),
                   'd8f717f8dd4b545cb7f418ef9f3d0c3e6e68a6f48b97d32f8b7aae40cb31f96f')
    ]

    _weight_files_2 = [
        WeightFile('par_vec_trained_400',
                   ('https://dl.dropboxusercontent.com/s/yn7n6eso6382ey9/par_vec_trained_400.tar.gz?dl=1'),
                   '8e7dc7f5876d764761a3093f6ddd315f295a3a6c8578efa078ad27baf08b2569'),
    ]

    ### Primitive Meta-data ###
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    _weights_configs_1 = LoadWeightsPrimitive._get_weight_installation(_weight_files_1)
    _weights_configs_2 = LoadWeightsPrimitive._get_weight_installation_tar(_weight_files_2)

    metadata = hyperparams.base.PrimitiveMetadata({
        "id": "b88b7cdd-f07c-4d56-a383-d0c46e2c056b",
        "version": config.VERSION,
        "name": "Semantic Type Inference",
        "description": "A primitive which detects semantic type of each column of data",
        "python_path": "d3m.primitives.schema_discovery.semantic_type.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.SCHEMA_DISCOVERY,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY]
        },
        "keywords": ['semantic type inference"', "data type detection", "data profiler"],
        "installation": [config.INSTALLATION] + _weights_configs_1 + _weights_configs_2,
    })

    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Union[typing.Dict[str, str], None]=None):
        super().__init__(hyperparams=hyperparams, volumes=volumes)
        self.volumes = volumes
        self.hyperparams = hyperparams

        # Intialize LoadWeightsPrimitive
        LoadWeightsPrimitive.__init__(self)

        # Import other needed modules
        LoadWeightsPrimitive._import_lib(self)

        self._weights_path = LoadWeightsPrimitive._get_weights_data_dir()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Returns output dataframe with the structural_type updated in the input metadata
        """
        ### User Variables ###
        nn_id     = 'sherlock'
        vec_dim   = 400
        n_samples = 1000

        # Load word vectors
        word_vec_path  = LoadWeightsPrimitive._find_weights_dir(key_filename='glove.6B.50d.txt', volumes=self.volumes)
        word_vectors_f = open(word_vec_path, encoding='utf-8')
        logging.info('Word vector loaded from: {}'.format(word_vec_path))

        # Load classes
        smi_cls_pth = LoadWeightsPrimitive._find_weights_dir(key_filename='classes_{}.npy'.format(nn_id), volumes=self.volumes)
        smi_classes = np.load(smi_cls_pth, allow_pickle=True)
        logging.info('Semantic Types loaded from: {}'.format(smi_cls_pth))

        # Load pretrained paragraph vector model -- Hard coded for now --assuiming loading from static file
        # par_vec_path = LoadWeightsPrimitive._find_weights_dir(key_filename='par_vec_trained_400', volumes=self.volumes)
        par_vec_path = os.path.join(self._weights_path, '8e7dc7f5876d764761a3093f6ddd315f295a3a6c8578efa078ad27baf08b2569/par_vec_trained_400/par_vec_trained_400.pkl')
        model = doc2vec.Doc2Vec.load(par_vec_path)
        logging.info('Pre-trained paragraph vector loaded from: {}'.format(par_vec_path))

        # Mapping dir of semantic types to D3M structural type dtypes of [int, str]
        smi_map_func = {'address':str, 'affiliate': str, 'affiliation': str,\
            'age':int, 'album':str, 'area':str, 'artist':str, 'birth Date':int,\
            'birth Place':str, 'brand':str, 'capacity':str, 'category':str,\
            'city':str, 'class':str, 'classification':str, 'club':str,\
            'code':str, 'collection':str, 'command':str, 'company':str,\
            'component':str, 'continent':str, 'country':str, 'county':str,\
            'creator':str, 'credit':str, 'currency':int, 'day':int, 'depth':int,\
            'description':str, 'director':str, 'duration':int, 'education':str,\
            'elevation':str, 'family':str, 'file Size':int, 'format':str,\
            'gender':str, 'genre':str, 'grades':str, 'industry':str, 'isbn':str,\
            'jockey':str, 'language':str, 'location':str, 'manufacturer':str,\
            'name':str, 'nationality':str, 'notes':str, 'operator':str,\
            'order':str, 'organisation':str, 'origin':str, 'owner':str,\
            'person':str, 'plays':str, 'position':int, 'product':str, 'publisher':str,\
            'range':str,  'rank':str, 'ranking':str, 'region':str, 'religion':str,\
            'requirement':str, 'result':str, 'sales':str, 'service':str, 'sex':str,\
            'species':str, 'state':str, 'status':str, 'symbol':str, 'team':str,\
            'team Name':str, 'type':str, 'weight':int, 'year':str}

        # print(smi_map_func)

        ### Build Features ###
        logging.info('Building Features in progress......')
        df_char = pd.DataFrame()
        df_word = pd.DataFrame()
        df_par  = pd.DataFrame()
        df_stat = pd.DataFrame()

        counter = 0

        if self.hyperparams['use_row_iter'] == True:
            for raw_sample in inputs.iterrows():
                # Extract data from series, if given as series list
                # Ex: 23 [95, 100, 95, 89, 84, 91, 88, 94, 75]
                #     45 [95, 100, 95, 89]
                if len(raw_sample) > 1:
                    raw_sample = literal_eval(raw_sample[1].loc[1])

                if counter % 1000 == 0:
                    logging.info('Completion {}/{}'.format(counter, len(inputs)))

                n_values = len(raw_sample)

                if n_samples > n_values:
                    n_samples = n_values

                # Sample n_samples from data column, and convert cell values to string values
                raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

                # Extract Features
                df_char = df_char.append(self._extract_bag_of_characters_features(raw_sample), ignore_index=True)
                df_word = df_word.append(self._extract_word_embeddings_features(word_vectors_f, raw_sample), ignore_index=True)
                df_par  = df_par.append(self._infer_paragraph_embeddings_features(model, raw_sample), ignore_index=True)
                df_stat = df_stat.append(self._extract_bag_of_words_features(raw_sample), ignore_index=True)

                # Increment the progress counter
                counter += 1

        else:

            for name, raw_sample in inputs.iteritems():
                #print(raw_sample)
                if counter % 1000 == 0:
                    logging.info('Completion {}/{}'.format(counter, len(inputs)))

                n_values = len(raw_sample)

                if n_samples > n_values:
                    n_samples = n_values

                # Sample n_samples from data column, and convert cell values to string values
                raw_sample = pd.Series(random.choices(raw_sample.tolist(), k=n_samples)).astype(str)

                # Extract Features
                df_char = df_char.append(self._extract_bag_of_characters_features(raw_sample), ignore_index=True)
                df_word = df_word.append(self._extract_word_embeddings_features(word_vectors_f, raw_sample), ignore_index=True)
                df_par  = df_par.append(self._infer_paragraph_embeddings_features(model, raw_sample), ignore_index=True)
                df_stat = df_stat.append(self._extract_bag_of_words_features(raw_sample), ignore_index=True)

                # Increment the progress counter
                counter += 1

        df_char.fillna(df_char.mean(), inplace=True)
        df_word.fillna(df_word.mean(), inplace=True)
        df_par.fillna(df_par.mean(),   inplace=True)
        df_stat.fillna(df_stat.mean(), inplace=True)

        logging.info('Completion {}/{}'.format(len(inputs), len(inputs)))

        # Collect all the features
        feature_vectors = [df_char.values, df_word.values, df_par.values, df_stat.values]

        # Free Memory
        word_vectors_f.close()
        del model

        logging.info('----------Feature Extraction Complete!-------------')

        ### Load Sherlock model ###
        sherlock_path = LoadWeightsPrimitive._find_weights_dir(key_filename='{}_model.json'.format(nn_id), volumes=self.volumes)
        file = open(sherlock_path, 'r')
        sherlock_file = file.read()
        sherlock = tf.keras.models.model_from_json(sherlock_file)
        file.close()

        # Load weights into new model
        sherlock_weights_path = LoadWeightsPrimitive._find_weights_dir(key_filename='{}_weights.h5'.format(nn_id), volumes=self.volumes)
        sherlock.load_weights(sherlock_weights_path)

        # Compile model
        sherlock.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
        # print(sherlock.summary())
        logging.info('------SMI Model Loaded Successfully!--------')

        ### Run Prediction ###
        y_pred = sherlock.predict(feature_vectors)
        # print('Prediction Completed!')
        y_pred_int = np.argmax(y_pred, axis=1)

        encoder = LabelEncoder()
        encoder.classes_ = smi_classes
        smi_preds = encoder.inverse_transform(y_pred_int)
        # print(smi_preds)

        ## Update structural_type of the input meta-data ###
        updated_types = []
        for smi_o in smi_preds:
            get_type = smi_map_func[smi_o]
            updated_types.append(get_type)

        # Outputs
        outputs = inputs

        # Get all columns and metadata
        columns_to_use = self._get_columns(inputs.metadata, str)
        outputs_metadata = inputs.metadata.select_columns(columns_to_use)

        # Update metadata for each column
        for col in range(len(columns_to_use)):
            outputs_metadata = outputs_metadata.update((columns_to_use[col], metadata_base.ALL_ELEMENTS), {
            'structural_type': updated_types[col],
            })

        outputs.metadata = outputs_metadata

        return base.CallResult(outputs)


    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int, type_to_cast: type) -> bool:
        # Always return true
        return True


    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata, type_to_cast: type) -> typing.Sequence[int]:
        # https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/cast_to_type.py
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index, type_to_cast)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            raise ValueError("No columns to be cast to type '{type}'.".format(type=type_to_cast))
        # We prefer if all columns could be cast, not just specified columns,
        # so we warn always when there are columns which cannot be produced.
        elif columns_not_to_use:
            self.logger.warning("Not all columns can be cast to type '%(type)s'. Skipping columns: %(columns)s", {
                'type': type_to_cast,
                'columns': columns_not_to_use,
            })

        return columns_to_use


    def _extract_word_embeddings_features(self, word_vectors_f, values: pd.DataFrame):
        """
        :param Data: (pandas series) A single column.
        # Output: ordered dictionary holding word embedding features
        """
        num_embeddings = 50
        f = OrderedDict()
        embeddings = []

        word_to_embedding = {}

        for w in word_vectors_f:
            term, vector = w.strip().split(' ', 1)
            vector = np.array(vector.split(' '), dtype=float)
            word_to_embedding[term] = vector

        values = values.dropna()

        for v in values:
            v = str(v).lower()

            if v in word_to_embedding:
                embeddings.append(word_to_embedding.get(v))
            else:
                words = v.split(' ')
                embeddings_to_all_words = []

                for w in words:
                    if w in word_to_embedding:
                        embeddings_to_all_words.append(word_to_embedding.get(w))
                if embeddings_to_all_words:
                    mean_of_word_embeddings = np.nanmean(embeddings_to_all_words, axis=0)
                    embeddings.append(mean_of_word_embeddings)

        if len(embeddings) == 0:
            for i in range(num_embeddings): f['word_embedding_avg_{}'.format(i)] = np.nan
            for i in range(num_embeddings): f['word_embedding_std_{}'.format(i)] = np.nan
            for i in range(num_embeddings): f['word_embedding_med_{}'.format(i)] = np.nan
            for i in range(num_embeddings): f['word_embedding_mode_{}'.format(i)] = np.nan

            f['word_embedding_feature'] = 0

            return f

        else:
            mean_embeddings = np.nanmean(embeddings, axis=0)
            med_embeddings  = np.nanmedian(embeddings, axis=0)
            std_embeddings  = np.nanstd(embeddings, axis=0)
            mode_embeddings = sy_stats.mode(embeddings, axis=0, nan_policy='omit')[0].flatten()

            for i, e in enumerate(mean_embeddings): f['word_embedding_avg_{}'.format(i)] = e
            for i, e in enumerate(std_embeddings): f['word_embedding_std_{}'.format(i)] = e
            for i, e in enumerate(med_embeddings): f['word_embedding_med_{}'.format(i)] = e
            for i, e in enumerate(mode_embeddings): f['word_embedding_mode_{}'.format(i)] = e

            f['word_embedding_feature'] = 1

            return f



    def _infer_paragraph_embeddings_features(self, model, data:pd.DataFrame):
        """
        :param Data: (pandas series) A single column.
        # Output: Ordered dictionary holding paragraph vector features
        """
        f = pd.DataFrame()

        if len(data) > 1000:
            vec = random.sample(data, 1000)
        else:
            vec = data

        # Infer paragraph vector for data sample
        f = f.append(pd.Series(model.infer_vector(vec, steps=20,
                                                  alpha=0.025)), ignore_index=True)

        col_names = []
        for i, col in enumerate(f):
            col_names.append('par_vec_{}'.format(i))

        f.columns = col_names

        return f



    def _extract_bag_of_characters_features(self, data:pd.DataFrame):
        """
        :param Data: (pandas series) A single column.
        # Output: Ordered dictionary holding bag of character features
        """
        characters_to_check = [ '['+ c + ']' for c in string.printable if c not in ( '\n', '\\', '\v', '\r', '\t', '^' )] + ['[\\\\]', '[\^]']

        f = OrderedDict()

        data_no_null = data.dropna()
        all_value_features = OrderedDict()

        for c in characters_to_check:
            all_value_features['n_{}'.format(c)] = data_no_null.str.count(c)

        for value_feature_name, value_features in all_value_features.items():
            f['{}-agg-any'.format(value_feature_name)] = any(value_features)
            f['{}-agg-all'.format(value_feature_name)] = all(value_features)
            f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
            f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
            f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
            f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
            f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
            f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
            f['{}-agg-kurtosis'.format(value_feature_name)] = sy_stats.kurtosis(value_features)
            f['{}-agg-skewness'.format(value_feature_name)] = sy_stats.skew(value_features)

        return f



    def _extract_bag_of_words_features(self, data:pd.DataFrame):
        """
        :param Data: (pandas series) A single column.
        # Output: Ordered dictionary holding bag of words features
        """
        f = OrderedDict()
        data = data.dropna()

        n_val = data.size

        if not n_val: return

        # Entropy of column
        freq_dist = nltk.FreqDist(data)
        probs = [freq_dist.freq(l) for l in freq_dist]
        f['col_entropy'] = -sum(p * math.log(p,2) for p in probs)

        # Fraction of cells with unique content
        num_unique = data.nunique()
        f['frac_unique'] = num_unique / n_val

        # Fraction of cells with numeric content -> frac text cells doesn't add information
        num_cells = np.sum(data.str.contains('[0-9]', regex=True))
        text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
        f['frac_numcells']  = num_cells / n_val
        f['frac_textcells'] = text_cells / n_val

        # Average + std number of numeric tokens in cells
        num_reg = '[0-9]'
        f['avg_num_cells'] = np.mean(data.str.count(num_reg))
        f['std_num_cells'] = np.std(data.str.count(num_reg))

        # Average + std number of textual tokens in cells
        text_reg = '[a-z]|[A-Z]'
        f['avg_text_cells'] = np.mean(data.str.count(text_reg))
        f['std_text_cells'] = np.std(data.str.count(text_reg))

        # Average + std number of special characters in each cell
        spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
        f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
        f['std_spec_cells'] = np.std(data.str.count(spec_reg))

        # Average number of words in each cell
        space_reg = '[" "]'
        f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
        f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

        all_value_features = OrderedDict()

        data_no_null = data.dropna()

        f['n_values'] = n_val

        all_value_features['length'] = data_no_null.apply(len)

        for value_feature_name, value_features in all_value_features.items():
            f['{}-agg-any'.format(value_feature_name)] = any(value_features)
            f['{}-agg-all'.format(value_feature_name)] = all(value_features)
            f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
            f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
            f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
            f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
            f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
            f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
            f['{}-agg-kurtosis'.format(value_feature_name)] = sy_stats.kurtosis(value_features)
            f['{}-agg-skewness'.format(value_feature_name)] = sy_stats.skew(value_features)

        n_none = data.size - data_no_null.size - len([ e for e in data if e == ''])
        f['none-agg-has'] = n_none > 0
        f['none-agg-percent'] = n_none / len(data)
        f['none-agg-num'] = n_none
        f['none-agg-all'] = (n_none == len(data))

        return f
