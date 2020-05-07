import d3m.metadata.base as metadata_module
from d3m import utils
from d3m.container.numpy import ndarray
from d3m.metadata import hyperparams, params
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, GradientCompositionalityMixin, Gradients

import os
import abc
import time
import numpy as np  # type: ignore
from operator  import mul
from functools import reduce
from typing    import Dict, List, Tuple, Type

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim  # type: ignore
from torch.autograd import Variable  # type: ignore

import pyro
from pyro.infer import SVI
from pyro.optim import ClippedAdam
import pyro.distributions as dist

from primitives.dmm.utils import to_variable
from primitives.dmm.dmm   import DMM, GaussianEmitter


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
    # Training parameters
    batch_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        default = 32,
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



class DeepMarkovModelPrimitive(GradientCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                               SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A deep markov model in the d3m interface
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
        "keywords": ['deep markov model', 'regression'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['learning_rate', 'minibatch_size']
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

        self._iterations_done = 0
        self._has_finished = False

        self._adam_params = {"lr": hyperparams['learning_rate'],\
                             "betas": (hyperparams['beta1'], hyperparams['beta2']),\
                             "clip_norm": hyperparams['clip_norm'],\
                             "lrd": hyperparams['lr_decay'],\
                             "weight_decay": hyperparams['weight_decay']}

        self._net = None  # type: Type[torch.nn.Module]

        # Is the model fit on data
        self._fitted = False

    def _create_dmm(self) -> Type[torch.nn.Module]:
        if not self._obs_dim:
            raise ValueError('cannot initialize the dmm without obs dim, set training data first')


        net = DMM(self._obs_dim, 'gaussian', self._latent_dim, self._emission_dim,\
                  self._transfer_dim, self._combiner_dim, self._rnn_dim, self._rnn_dropout_rate)

        return net


    def _curate_data(self, training_inputs, training_outputs, get_labels):
        # if self._training_inputs is None or self._training_outputs is None:
        if training_inputs is None:
            raise ValueError("Missing data.")

        # Get training data and labels data
        try:
            feature_columns_1 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        except:
            feature_columns_1 = None
        try:
            feature_columns_2 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        except:
            feature_columns_2 = None
        # Remove columns if outputs present in inputs
        if len(feature_columns_2) >= 1:
            for fc_2 in feature_columns_2:
                try:
                    feature_columns_1.remove(fc_2)
                except ValueError:
                    pass

        # Get labels data if present in training input
        try:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        except:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # If no error but no label-columns found, force try SuggestedTarget
        if len(label_columns) == 0 or label_columns == None:
            label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        # Remove columns if outputs present in inputs
        if len(label_columns) >= 1:
            for lbl_c in label_columns:
                try:
                    feature_columns_1.remove(lbl_c)
                except ValueError:
                    pass

        # Training Set
        feature_columns_1 = [int(fc) for fc in feature_columns_1]
        try:
            new_XTrain = ((training_inputs.iloc[:, feature_columns_1]).to_numpy()).astype(np.float)
        except ValueError:
            # Most likely Numpy ndarray series
            XTrain = training_inputs.iloc[:, feature_columns_1]
            XTrain_shape = XTrain.shape[0]
            XTrain = ((XTrain.iloc[:, -1]).to_numpy())
            # Unpack
            new_XTrain = []
            for arr in range(XTrain_shape):
                new_XTrain.append(XTrain[arr])

            new_XTrain = np.array(new_XTrain)

            # del to save memory
            del XTrain

        # Training labels
        if get_labels:
            if training_outputs is None:
                raise ValueError("Missing data.")

            # Get label column names
            label_name_columns  = []
            label_name_columns_ = list(training_outputs.columns)
            for lbl_c in label_columns:
                label_name_columns.append(label_name_columns_[lbl_c])

            self.label_name_columns = label_name_columns

            # Get labelled dataset
            try:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            except ValueError:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            # If no error but no label-columns force try SuggestedTarget
            if len(label_columns) == 0 or label_columns == None:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            YTrain = ((training_outputs.iloc[:, label_columns]).to_numpy()).astype(np.float)

            return new_XTrain, YTrain, feature_columns_1

        return new_XTrain, feature_columns_1


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        inputs, outputs, _ = self._curate_data(training_inputs=inputs, training_outputs=outputs, get_labels=True)

        if len(inputs) != len(outputs):
            raise ValueError('Training data sequences "inputs" and "outputs" should have the same length.')

        self._training_size    = len(inputs)
        self._training_inputs  = to_variable(inputs)
        self._training_outputs = to_variable(outputs)

        while len(self._training_outputs.shape) < 3:
            self._training_outputs = self._training_outputs.unsqueeze(1)
        while len(self._training_inputs.shape) < 3:
            self._training_inputs = self._training_inputs.unsqueeze(1)

        # Expects this to be in num_sequences x num_timesteps x obs_dim
        self._training_data = torch.cat((self._training_inputs, self._training_outputs), dim=1)

        self._obs_dim    = self._training_data.shape[-1]
        self._seq_length = self._training_data.shape[1]

        self._net = self._create_dmm()

        adam = ClippedAdam(self._adam_params)
        self._optimizer       = SVI(self._net.model, self._net.guide, adam, "ELBO", trace_graph=False)
        self._iterations_done = 0
        self._has_finished    = False


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._net is None:
            raise Exception('Neural network not initialized. You need to set training data so that the network structure can be defined.')

        # Set model to eval
        self._net.eval()

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

    def _reverse_sequences_numpy(self, mini_batch: ndarray) -> ndarray:
        reversed_mini_batch = mini_batch.copy()
        for b in range(mini_batch.shape[0]):
            T = self._seq_length
            reversed_mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
        return reversed_mini_batch

    def _get_mini_batch(self, mini_batch_indices: List, sequences: ndarray) -> Tuple[ndarray, ndarray]:
        mini_batch = sequences[mini_batch_indices, :, :]
        mini_batch_reversed = self._reverse_sequences_numpy(mini_batch)

        # Wrap in PyTorch Variables
        mini_batch = Variable(torch.Tensor(mini_batch))
        mini_batch_reversed = Variable(torch.Tensor(mini_batch_reversed))

        return mini_batch, mini_batch_reversed

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None:
            raise Exception('Cannot fit when no training data is present.')

        if self._fitted:
            return base.CallResult(None)

        if timeout is None:
            timeout = np.inf
        if iterations is None:
            iterations = 100

        N_train_data = self._training_data.shape[0]
        mini_batch_size = self._batch_size
        N_mini_batches = int(N_train_data / mini_batch_size + int(N_train_data % mini_batch_size > 0))

        start = time.time()
        self._iterations_done = 0
        # We can always iterate more, even if not reasonable.
        self._has_finished = False
        self._net.train()
        times = [time.time()]
        iteration_count = 0
        epoch = 0

        while(True):
            epoch_nll = 0.0
            shuffled_indices = np.arange(N_train_data)
            np.random.shuffle(shuffled_indices)
            epoch += 1
            for which_mini_batch in range(N_mini_batches):
                mini_batch_start = (which_mini_batch * mini_batch_size)
                mini_batch_end = np.min([(which_mini_batch + 1) * mini_batch_size, N_train_data])
                mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]

                mini_batch, mini_batch_reversed = self._get_mini_batch(mini_batch_indices, self._training_data.data.numpy())
                should = False

                # Just give minibatch and minibatch reversed
                loss = self._optimizer.step(mini_batch, mini_batch_reversed)
                iteration_count += 1
                epoch_nll += loss

                if np.isnan(loss):
                    print("minibatch nan-ed out!")
                    break
                if np.isinf(loss):
                    print("minibatch inf-ed out!")
                    break
                if iteration_count == iterations:
                    break

            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            print("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" % (epoch, epoch_nll / 20, epoch_time))
            if iteration_count == iterations:
                break

        self._iterations_done += iteration_count
        self._fitted = True

        return CallResult(None)

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
