from d3m import utils
from d3m import container
from d3m.container.numpy import ndarray
import d3m.metadata.base as metadata_module
from d3m.metadata import hyperparams, params
from d3m.metadata import base as metadata_base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.base import Gradients
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin
from d3m.primitive_interfaces.base import GradientCompositionalityMixin
from d3m.primitive_interfaces.base import SamplingCompositionalityMixin

import os
import abc
import time
import torch  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import mean_squared_error
from typing import NamedTuple, Sequence, Any, List, Dict, Union, Tuple

# Import config file
from primitives.config_files import config
from primitives.diagonalMVN.utils import to_variable, refresh_node, log_mvn_likelihood

__all__ = ('DiagonalMVNPrimitive',)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    mean: ndarray
    covariance: ndarray

class Hyperparams(hyperparams.Hyperparams):
    alpha = hyperparams.Hyperparameter[float](
        default=1e-2,
        description='initial fitting step size',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    beta = hyperparams.Hyperparameter[float](
        default=1e-8,
        description='see reference for details',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=1000,
        description='if there is a lot of data, primitive fits using gradient descent in which case, specify batch size here',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    num_iterations = hyperparams.Hyperparameter[int](
        default=100,
        description="Number of iterations to sample the model.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class DiagonalMVNPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                           GradientCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                           SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                           SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive which allows fitting, and sampling from, a multivariate Gaussian (with diagonal covariance matrix)
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame of features of shape: NxM, where N = samples and M = features.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "d211d6ca-4a31-4c69-a853-3be9c57fa36f",
        "version": config.VERSION,
        "name": "Diagonal Multivariate Normal Distribution primitive",
        "description": "Diagonal multivariate normal distribution primitive, mainly for being used as a weight prior of another primitive.",
        "python_path": "d3m.primitives.operator.DiagonalMVN.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.OPERATOR,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NORMAL_DISTRIBUTION,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['normal', 'distribution'],
        "installation": [config.INSTALLATION],
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs:  Inputs  = None
        self._training_outputs: Outputs = None

        self._alpha = hyperparams['alpha']
        self._beta  = hyperparams['beta']
        self._batch_size = hyperparams['batch_size']

        self._mean = None
        self._covariance = None
        self._iterations_done = None
        self._fit_term_temperature = 0.0

        # Is the model fit on data
        self._fitted = False


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs   = inputs
        self._training_outputs  = None
        self._new_training_data = True


    def _curate_data(self, training_inputs, training_outputs, get_labels):
        # if self._training_inputs is None or self._training_outputs is None:
        if training_inputs is None:
            raise ValueError("Missing data.")

        # Get training data and labels data
        try:
            feature_columns_1 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Attribute')
        except:
            feature_columns_1 = []
        try:
            feature_columns_2 = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        except:
            feature_columns_2 = []
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
                XTrain_Flatten = (XTrain[arr]).flatten()
                new_XTrain.append(XTrain_Flatten)
            new_XTrain = np.array(new_XTrain)

        # Get label column names
        label_name_columns  = []
        label_name_columns_ = list(training_inputs.columns)
        for lbl_c in label_columns:
            label_name_columns.append(label_name_columns_[lbl_c])

        if get_labels:
            # Training labels
            YTrain = np.array([])
            # Get labelled dataset if available
            try:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            except ValueError:
                label_columns  = training_outputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            # If no error but no label-columns force try SuggestedTarget
            if len(label_columns) == 0 or label_columns == []:
                label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
            if len(label_columns) > 0:
                try:
                    YTrain = ((training_inputs.iloc[:, label_columns]).to_numpy()).astype(np.int)
                except:
                    # Maybe no labels or missing labels
                    YTrain = (training_inputs.iloc[:, label_columns].to_numpy())

            return new_XTrain, YTrain, feature_columns_1, label_name_columns

        return new_XTrain, feature_columns_1, label_name_columns


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None:
            raise Exception('Cannot fit when no training data is present.')

        if self._fitted:
            return base.CallResult(None)

        if timeout is None:
            timeout = np.inf

        # Set all files
        if iterations == None:
            _iterations = self.hyperparams['num_iterations']
        else:
            _iterations = iterations

        # Curate data
        XTrain, _, _ = self._curate_data(training_inputs=self._training_inputs, training_outputs=None, get_labels=False)

        self._training_outputs = to_variable(XTrain, requires_grad=True)
        # if its low dimensional and not much data
        if sum(self._training_outputs.shape) < 10000:
            nd = self._training_outputs.data.numpy()

            self._mean = torch.Tensor(np.average(nd, axis=0))
            self._covariance = torch.Tensor(np.diag(np.square(np.std(nd, axis=0))))
            self._iterations_done = iterations
            self._new_training_outputs = False

            return CallResult(None)

        # Initializing fitting from scratch.
        if self._new_training_outputs:
            self._mean = to_variable(np.zeros(self._training_outputs.size()[0]), True)
            self._covariance = to_variable(np.eye(self._training_outputs.size()[0]), True)
            self._new_training_outputs = False


        start = time.time()
        # We can always iterate more, even if not reasonable.
        self._fitted = False
        self._iterations_done = 0
        prev_mean_grad, prev_covariance_grad = None, None  # type: torch.Tensor, torch.Tensor

        while time.time() < start + timeout and self._iterations_done < iterations:
            self._iterations_done += 1

            data_count = len(self._training_outputs)
            if data_count > self._batch_size:
                sample = self._training_outputs.data.numpy()[np.random.choice(data_count, self._batch_size), :]
            else:
                sample = self._training_outputs

            log_likelihood = sum(self._log_likelihood(output=training_output)
                                 for training_output in sample)

            self._mean.retain_grad()
            self._covariance.retain_grad()
            log_likelihood.backward()

            mean_grad = self._mean.grad.data
            covariance_grad = self._covariance.grad.data

            if prev_mean_grad is not None:
                self._alpha += self._beta * \
                              (torch.dot(mean_grad, prev_mean_grad)
                               + torch.dot(covariance_grad.view(-1),
                                           prev_covariance_grad.view(-1)))
            prev_mean_grad, prev_covariance_grad = mean_grad, covariance_grad

            self._mean.data += mean_grad * self._alpha / torch.norm(mean_grad)
            self._covariance.data += covariance_grad * self._alpha / torch.norm(prev_covariance_grad)

            self._mean = refresh_node(self._mean)
            self._covariance = refresh_node(self._covariance)

        self._fitted = True

        return CallResult(None)


    # Our best choice is mean.
    def _produce_one(self) -> ndarray:
        return self._mean.data.numpy()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._mean is None:
            raise ValueError("Missing parameter 'mean' - call 'fit' first")

        # Curate data
        XTest, _, _ = self._curate_data(training_inputs=inputs, training_outputs=None, get_labels=False)

        self._fitted = True
        self._iterations_done = None
        predictions = np.array([self._produce_one() for _ in XTest])
        # Convert from ndarray from DataFrame
        predictions = container.DataFrame(predictions, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(predictions.shape[1]):
            col_dict = dict(predictions.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = 'feature_'+str(col)
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
            predictions.metadata        = predictions.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

        return CallResult(predictions, has_finished=False, iterations_done=self._iterations_done)


    def backward(self, *, gradient_outputs: Gradients[Outputs], fine_tune: bool = False, fine_tune_learning_rate: float = 0.00001, fine_tune_weight_decay: float = 0.00001) -> Tuple[Gradients[Inputs], Gradients[Params]]: # type: ignore
        raise NotImplementedError('mvn does not support backward')

    def get_params(self) -> Params:
        return Params(mean=ndarray(self._mean.data.numpy()), covariance=ndarray(self._covariance.data.numpy()))

    def set_params(self, *, params: Params) -> None:
        self._mean = to_variable(params['mean'], requires_grad=True)
        self._covariance = to_variable(params['covariance'], requires_grad=True)

    def _sample_once(self, *, inputs: Inputs) -> Outputs:
        mean = self._mean.data.numpy()
        covariance = self._covariance.data.numpy()

        return np.array([np.random.multivariate_normal(mean, covariance) for _ in inputs])

    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> Sequence[Outputs]:
        # sample just returns a number of samples from the current mvn
        s = np.array([self._sample_once(inputs=inputs) for _ in range(num_samples)])

        return CallResult(s, has_finished=False, iterations_done=self._iterations_done)

    def _log_likelihood(self, *, output:  torch.autograd.Variable) -> torch.autograd.Variable:
        """
        Calculates log(normal_density(self._mean, self._covariance)).
        """
        output = to_variable(output)

        return log_mvn_likelihood(self._mean, self._covariance, output)

    def _gradient_output_log_likelihood(self, *, output:  torch.autograd.Variable) -> torch.autograd.Variable:
        """
        Output is D-length torch variable.
        """
        output = refresh_node(output)
        log_likelihood = self._log_likelihood(output=output)
        log_likelihood.backward()

        return output.grad

    def _gradient_params_log_likelihood(self, *, output:  torch.autograd.Variable) -> Tuple[torch.autograd.Variable, torch.autograd.Variable]:
        """
        Output is D-length torch variable.
        """
        self._mean = refresh_node(self._mean)
        self._covariance = refresh_node(self._covariance)
        log_likelihood = self._log_likelihood(output=output)
        log_likelihood.backward()

        return (self._mean.grad, self._covariance.grad)

    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[ndarray]:
        result = np.array([self._log_likelihood(output=output).data.numpy() for output in outputs])

        return CallResult(result, has_finished=False, iterations_done=self._iterations_done)

    def log_likelihood(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[float]:
        """
        Calculates log(normal_density(self._mean, self._covariance)).
        """
        result = self.log_likelihoods(outputs=outputs, inputs=inputs, timeout=timeout, iterations=iterations)

        return CallResult(sum(result.value), has_finished=result.has_finished, iterations_done=result.iterations_done)

    def gradient_output(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Outputs]:  # type: ignore
        """
        Calculates gradient of log(normal_density(self._mean, self._covariance)) * fit_term_temperature with respect to output.
        """

        outputs_vars = [to_variable(output, requires_grad=True) for output in outputs]

        grad = sum(self._gradient_output_log_likelihood(output=output)
                   for output in outputs_vars)

        return grad.data.numpy()

    def gradient_params(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Params]:  # type: ignore
        """
        Calculates gradient of log(normal_density(self._mean, self._covariance)) * fit_term_temperature with respect to params.
        """
        outputs_vars = [to_variable(output, requires_grad=True) for output in outputs]

        grads = [self._gradient_params_log_likelihood(output=output)
                 for output in outputs_vars]
        grad_mean = sum(grad[0] for grad in grads)
        grad_covariance = sum(grad[1] for grad in grads)

        return Params(mean=ndarray(grad_mean.data.numpy()), covariance=ndarray(grad_covariance.data.numpy()))

    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        self._fit_term_temperature = temperature

    def get_call_metadata(self) -> CallResult:
        return CallResult(None, has_finished=False, iterations_done=self._iterations_done)
