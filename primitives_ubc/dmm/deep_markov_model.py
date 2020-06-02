import d3m.metadata.base as metadata_module
from d3m import utils
from d3m import container
from d3m.container.numpy import ndarray
from d3m.metadata import hyperparams, params
from d3m.metadata import base as metadata_base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, GradientCompositionalityMixin, Gradients

import os
import abc
import time
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from operator  import mul
from functools import reduce
from typing    import Dict, List, Tuple, Type

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils import data # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim  # type: ignore
from torch.autograd import Variable  # type: ignore

import pyro
from pyro.infer import SVI
from pyro.optim import ClippedAdam
import pyro.distributions as dist

from primitives_ubc.dmm.utils import to_variable
from primitives_ubc.dmm.dmm   import DMM, GaussianEmitter

# Import config file
from primitives_ubc.config_files import config

from primitives_ubc.dmm.dataset import Dataset

__all__ = ('DeepMarkovModelPrimitive',)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    state: Dict


class Hyperparams(hyperparams.Hyperparams):
    latent_dim = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 50,
        description='latent dimension'
    )
    emission_dim = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 50,
        description='emissio dimension'
    )
    transfer_dim = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 30,
        description='transfer dimension'
    )
    combiner_dim = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 50,
        description='combiner dimension'
    )
    rnn_dim = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 200,
        description='rnn proposal dimension'
    )
    rnn_dropout_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 0.0,
        description='dropout when training rnn'
    )
    seq_length = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default = 15,
        description='sequence length of the RNN'
    )
    # Training parameters
    batch_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 3,
        description='batch size for training'
    )
    learning_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 0.00004,
        description='learning rate to use for training'
    )
    beta1 = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 0.96,
        description='beta1 used in adam'
    )
    beta2 = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 0.999,
        description='beta2 used in adam'
    )
    clip_norm = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 20.0,
        description='gradient clipping'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 0.99996,
        description='lr decay in adam'
    )
    weight_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 0.6,
        description='weight decay'
    )
    predict_samples = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 10,
        description='number of samples to use in produce'
    )
    num_iterations = hyperparams.Hyperparameter[int](
        default=100,
        description="Number of iterations to train the model.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class DeepMarkovModelPrimitive(GradientCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                               SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A deep markov model in the d3m interface implemented in PyTorch
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "f59200c3-f597-4c92-9793-c2664e6932f8", # change
        "version": config.VERSION,
        "name": "Deep Markov Model Primitive",
        "description": "Deep Markov Model using Pytorch Framework",
        "python_path": "d3m.primitives.regression.DeepMarkovModel.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NEURAL_NETWORK_BACKPROPAGATION,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['deep markov model', 'regression', 'time series forecasting'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['batch_size', 'learning_rate', 'clip_norm', 'lr_decay', 'beta1', 'beta2', 'weight_decay', 'predict_samples']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs:  Inputs  = None
        self._training_outputs: Outputs = None

        self._latent_dim       = hyperparams['latent_dim']
        self._emission_dim     = hyperparams['emission_dim']
        self._transfer_dim     = hyperparams['transfer_dim']
        self._combiner_dim     = hyperparams['combiner_dim']
        self._batch_size       = hyperparams['batch_size']
        self._rnn_dim          = hyperparams['rnn_dim']
        self._predict_samples  = hyperparams['predict_samples']
        self._rnn_dropout_rate = hyperparams['rnn_dropout_rate']
        self._num_iterations   = hyperparams['num_iterations']

        self._has_finished = False
        self._constant = 1  # the constant term to avoid nan

        self._adam_params = {"lr": hyperparams['learning_rate'],\
                             "betas": (hyperparams['beta1'], hyperparams['beta2']),\
                             "clip_norm": hyperparams['clip_norm'],\
                             "lrd": hyperparams['lr_decay'],\
                             "weight_decay": hyperparams['weight_decay']}

        self._net = None  # type: Type[torch.nn.Module]

        # Is the model fit on data
        self._fitted = False

        # Use GPU if available
        use_cuda    = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.use_cuda = use_cuda


    def _create_dmm(self) -> Type[torch.nn.Module]:
        if not self._obs_dim:
            raise ValueError('cannot initialize the dmm without obs dim, set training data first')

        net = DMM(self._obs_dim, 'gaussian', self._latent_dim, self._emission_dim,\
                  self._transfer_dim, self._combiner_dim, self._rnn_dim,\
                  self._rnn_dropout_rate, use_cuda=self.use_cuda)

        return net


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        data = inputs.horizontal_concat(outputs)
        data = data.copy()

        # mark datetime column
        times = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        if len(times) != 1:
            raise ValueError(
                f"There are {len(times)} indices marked as datetime values. Please only specify one"
            )
        self._time_column = list(data)[times[0]]

        # if datetime columns are integers, parse as # of days
        if (
                "http://schema.org/Integer"
                in inputs.metadata.query_column(times[0])["semantic_types"]
        ):
            self._integer_time = True
            data[self._time_column] = pd.to_datetime(
                data[self._time_column] - 1, unit="D"
            )
        else:
            data[self._time_column] = pd.to_datetime(
                data[self._time_column], unit="s"
            )

        # sort by time column
        data = data.sort_values(by=[self._time_column])

        # mark key and grp variables
        self.key = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )

        # mark target variables
        self._targets = data.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        self._target_types = [
            "i"
            if "http://schema.org/Integer"
               in data.metadata.query_column(t)["semantic_types"]
            else "c"
            if "https://metadata.datadrivendiscovery.org/types/CategoricalData"
               in data.metadata.query_column(t)["semantic_types"]
            else "f"
            for t in self._targets
        ]
        self._targets = [list(data)[t] for t in self._targets]

        self.target_column = self._targets[0]

        # see if 'GroupingKey' has been marked
        # otherwise fall through to use 'SuggestedGroupingKey'
        grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = data.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
            drop_list = []
        else:
            drop_list = suggested_grouping_keys

        grouping_keys_counts = [
            data.iloc[:, key_idx].nunique() for key_idx in grouping_keys
        ]
        grouping_keys = [
            group_key
            for count, group_key in sorted(zip(grouping_keys_counts, grouping_keys))
        ]
        self.filter_idxs = [list(data)[key] for key in grouping_keys]

        # drop index
        data.drop(
            columns=[list(data)[i] for i in drop_list + self.key], inplace=True
        )

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([data[self._time_column], data[self.target_column]], axis=1)
            concat.columns = ['ds', 'y']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = data.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat,
                                data[self._time_column],
                                data[self.target_column]],
                               axis=1)
            concat.columns = ['unique_id', 'ds', 'y']

        # Series must be complete in the frequency
        concat = DeepMarkovModelPrimitive._ffill_missing_dates_per_serie(concat, 'D')

        # remove duplicates
        concat = concat.drop_duplicates(['unique_id', 'ds'])

        self._training_inputs = concat


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None:
            raise Exception('Cannot fit when no training data is present.')

        if self._fitted:
            return base.CallResult(None)

        # Extract curated data into X and Y's
        X_train = self._training_inputs[['unique_id', 'ds']]
        X_train['x'] = '1'
        y_train = self._training_inputs[['unique_id', 'ds', 'y']]
        y_train['y'] += self._constant # To remove missing values

        if timeout is None:
            timeout = np.inf
        if iterations is None:
            _iterations = self._num_iterations
        else:
            _iterations = iterations

        # Dataloader
        training_set = Dataset(config=self.hyperparams, X=X_train, y=y_train, min_series_length=self.hyperparams['seq_length'])
        # If the length is less than hyperparams, defaults to the minimum in dataset
        self._seq_length = min(self.hyperparams['seq_length'], training_set.min_series_length)

        # Dataset Parameters
        train_params = {'batch_size': self._batch_size,
                        'shuffle': True}

        # Data Generators
        training_generator = data.DataLoader(training_set, **train_params)

        # Setup Model
        self._obs_dim   = training_set.n_series
        self._net       = self._create_dmm()
        adam            = ClippedAdam(self._adam_params)
        self._optimizer = SVI(self._net.model, self._net.guide, adam, "ELBO")

        # Train functions
        self._iterations_done = 0
        self._has_finished    = False

        # Set model to training
        self._net.train()

        # for iters in _iterations:
        #     epoch_nll = 0.0
        #     iteration_count = 0
        #     for local_batch, local_labels in training_generator:
        #         _local_training_batch = torch.cat((local_batch, local_labels), axis=2)
        #         mini_batch, mini_batch_reversed = self._reverse_sequences(mini_batch=_local_training_batch)
        #         # Loss for minibatch and minibatch reversed
        #         loss = self._optimizer.step(mini_batch, mini_batch_reversed)
        #         iteration_count += 1
        #         epoch_nll += loss
        #         # Break
        #         if np.isnan(loss):
        #             print("minibatch nan-ed out!")
        #             break
        #         if np.isinf(loss):
        #             print("minibatch inf-ed out!")
        #             break
        #
        #     print("[training epoch %04d]  %.4f " % (epoch, epoch_nll/iteration_count))

        self._fitted = True

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._net is None:
            raise Exception('Neural network not initialized. You need to set training data so that the network structure can be defined.')

        if self._fitted is None:
            raise Exception('Neural network not Fitted. Please fit the model first.')

        # Get testing data
       inputs_copy = inputs.copy()
        # if datetime columns are integers, parse as # of days
        if self._integer_time:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column] - 1, unit="D"
            )
        else:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column], unit="s"
            )

        # find marked 'GroupingKey' or 'SuggestedGroupingKey'
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
        else:
            inputs_copy = inputs_copy.drop(columns=[list(inputs_copy)[i] for i in suggested_grouping_keys])

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:
            concat = pd.concat([inputs_copy[self._time_column]], axis=1)
            concat.columns = ['ds']
            concat['unique_id'] = 'series1'  # We have only one series
        else:
            # concatenate columns in `grouping_keys` to unique_id column
            concat = inputs_copy.loc[:, self.filter_idxs].apply(lambda x: ' '.join([str(v) for v in x]), axis=1)
            concat = pd.concat([concat, inputs_copy[self._time_column]], axis=1)
            concat.columns = ['unique_id', 'ds']

        # Final testing data
        X_test = concat[['unique_id', 'ds']]

        # Set model to eval
        self._net.eval()

        # Dataloader
        testing_set = Dataset(config=self.hyperparams, X=X_train, y=y_train, min_series_length=self._seq_length)

        self._input = to_variable(inputs)
        while len(self._input.shape) < 3:
            self._input = self._input.unsqueeze(0)
        input_reversed = self._reverse_sequences_numpy(self._input.data.numpy())

        # Wrap in PyTorch Variables
        input_reversed = Variable(torch.Tensor(input_reversed))

        posterior = pyro.infer.Importance(self._net.model, self._net.guide, num_samples=self._predict_samples)
        posterior_trace_generator = posterior._traces(self._input, input_reversed)

        log_weights = 0
        log_weighted_samples = 0

        for i in range(self._predict_samples):
            (model_trace, weight) = next(posterior_trace_generator)
            sampled_z = model_trace.nodes["_RETURN"]["value"]
            log_weighted_samples += sampled_z * weight[0]
            log_weights += weight[0]

        z_t = log_weighted_samples / log_weights
        z_mu, z_sigma = self._net.trans(z_t)
        z_tp1 = pyro.sample("z_t_plus_1", dist.normal, z_mu, z_sigma)

        emission_mus_t, emission_sigmas_t = self._net.emitter(z_tp1)
        zeros   = Variable(torch.zeros(emission_mus_t.size()))
        timings = pyro.sample("obs_t_plus_1", dist.normal, emission_mus_t, emission_sigmas_t)

        self._input = timings


        return timings


    def _reverse_sequences(self, mini_batch):
        reversed_mini_batch = mini_batch.clone()
        T = self._seq_length
        for b in range(mini_batch.shape[0]):
            reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1):, :]

        return mini_batch, reversed_mini_batch

    @staticmethod
    def _ffill_missing_dates_particular_serie(serie, min_date, max_date, freq):
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        unique_id = serie['unique_id'].unique()
        df_balanced = pd.DataFrame({'ds': date_range, 'key': [1] * len(date_range), 'unique_id': unique_id[0]})

        # Check balance
        check_balance = df_balanced.groupby(['unique_id']).size().reset_index(name='count')
        assert len(set(check_balance['count'].values)) <= 1
        df_balanced = df_balanced.merge(serie, how="left", on=['unique_id', 'ds'])

        df_balanced['y'] = df_balanced['y'].fillna(method='ffill')

        return df_balanced


    @staticmethod
    def _ffill_missing_dates_per_serie(df, freq="D", fixed_max_date=None):
        """
        Receives a DataFrame with a date column and forward fills the missing
        gaps in dates, not filling dates before the first appearance of a unique key

        Parameters
        ----------
        df: DataFrame
            Input DataFrame
        key: str or list
            Name(s) of the column(s) which make a unique time series
        date_col: str
            Name of the column that contains the time column
        freq: str
            Pandas time frequency standard strings, like "W-THU" or "D" or "M"
        numeric_to_fill: str or list
            Name(s) of the columns with numeric values to fill "fill_value" with
        """
        if fixed_max_date is None:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min', 'max']).reset_index()
        else:
            df_max_min_dates = df[['unique_id', 'ds']].groupby('unique_id').agg(['min']).reset_index()
            df_max_min_dates['max'] = fixed_max_date

        df_max_min_dates.columns = df_max_min_dates.columns.droplevel()
        df_max_min_dates.columns = ['unique_id', 'min_date', 'max_date']

        df_list = []
        for index, row in df_max_min_dates.iterrows():
            df_id = df[df['unique_id'] == row['unique_id']]
            df_id = DeepMarkovModelPrimitive._ffill_missing_dates_particular_serie(df_id, row['min_date'],
                                                                                    row['max_date'], freq)
            df_list.append(df_id)

        df_dates = pd.concat(df_list).reset_index(drop=True).drop('key', axis=1)[['unique_id', 'ds', 'y']]

        return df_dates

    def _fillna(self, series):
        if series.isnull().any():
            return series.fillna(self._data['y'].mean())
        return series

    def _get_params(self, state_dict) -> Params:
        s = {k: v.numpy() for k, v in state_dict.items()}
        return Params(state=s)

    def get_params(self) -> Params:
        return self._get_params(self._net.state_dict())

    def set_params(self, *, params: Params) -> None:
        state = self._net.state_dict()
        new_state = {k: torch.from_numpy(v) for k, v in params['state'].items()}
        state.update(new_state)
        self._net.load_state_dict(state)

    def get_call_metadata(self) -> CallResult:
        return CallResult(None, has_finished=self._has_finished, iterations_done=self._iterations_done)

    def backward(self, *, gradient_outputs: Gradients[Outputs], fine_tune: bool = False, fine_tune_learning_rate: float = 0.00001,
                 fine_tune_weight_decay: float = 0.00001) -> Tuple[Gradients[Inputs], Gradients[Params]]:  # type: ignore
        raise NotImplementedError()

    def gradient_output(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Outputs]:
        raise NotImplementedError()

    def gradient_params(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Params]:
        raise NotImplementedError()

    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        raise NotImplementedError()
