from d3m.container.numpy import ndarray
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin
from d3m.primitive_interfaces.base import GradientCompositionalityMixin
from d3m.primitive_interfaces.base import SamplingCompositionalityMixin
from d3m.primitive_interfaces.base import Gradients


import os
import math
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import NamedTuple, Sequence, Any, List, Dict, Union, Tuple
from .utils import to_variable, refresh_node, log_mvn_likelihood


__all__ = ('LinearRegressionPrimitive',)

Inputs  = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    weights: ndarray
    weights_variance: ndarray
    offset: float
    noise_variance: float


class Hyperparams(hyperparams.Hyperparams):
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        description='Learning rate used during training (fit).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    weights_prior = hyperparams.Hyperparameter[Union[None, GradientCompositionalityMixin]](
        default=None,
        description='prior on weights',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tune_prior_end_to_end = hyperparams.Hyperparameter[int](
        default=False,
        description='setting this to true will case the end to end training to propogate \
                    back to the prior parameters. what this means is that the actual \
                    parameters of the prior distribution are going to be changed\
                    according to the chain rule.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    analytic_fit_threshold = hyperparams.Hyperparameter[int](
        default=100,
        description='the threshold used for N/P where n is the number of training data, \
                    and P is the number of features. The training matrix X is likely to be \
                    rank deficient when N >> P or when P > N. If these are both not the case \
                    i.e. the matrix is full rank, we can use the analytic version. \
                    You can also force this primitive to use gradient optimization \
                    by setting this to zero',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class LinearRegressionPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                GradientCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "f59200c3-f597-4c92-9793-c2664e6932f8",
        "version": config.VERSION,
        "name": "Bayesian Linear Regression",
        "description": "A bayesian linear regression",
        "python_path": "d3m.primitives.regression.LinearRegression.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LINEAR_REGRESSION,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['bayesian', 'regression'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": []
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None

        self._learning_rate          = hyperparams['learning_rate']
        self._analytic_fit_threshold = hyperparams['analytic_fit_threshold']
        self._weights_prior          = hyperparams['weights_prior']
        self._tune_prior_end_to_end  = hyperparams['tune_prior_end_to_end']

        self._fit_term_temperature = 0.0
        self._weights              = None  # type: torch.autograd.Variable
        self._noise_variance       = None
        self._weights_variance     = None
        self._iterations_done      = None  # type: int
        self._has_finished         = False
        self._new_training_data    = True
        self._inputs               = None
        self._outputs              = None
        self._use_analytic_form    = False


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
            YTrain = ((training_outputs.iloc[:, label_columns]).to_numpy()).astype(np.int)

            return new_XTrain, YTrain, feature_columns_1

        return new_XTrain, feature_columns_1


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        inputs, outputs, _ = self._curate_data(training_inputs=inputs, training_outputs=outputs, get_labels=True)

        N, P = inputs.shape
        if P < N and N/P < self._analytic_fit_threshold:
            # TODO if P is less than N without regularization give user warning and try to add a default one
            self._use_analytic_form = True

        inputs_with_ones = np.insert(inputs, P, 1, axis=1)

        self._training_inputs   = to_variable(inputs_with_ones, requires_grad=True)
        self._training_outputs  = to_variable(outputs, requires_grad=True)
        self._new_training_data = True
        self._has_finished      = False
        self._iterations_done   = 0
        self._converged_count   = 0
        self._best_rmse         = np.inf


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        inputs : (num_inputs,  D) numpy array
        outputs : numpy array of dimension (num_inputs)
        """

        inputs = self._offset_input(inputs=inputs)

        self._weights = refresh_node(self._weights)
        self._noise_variance = refresh_node(self._noise_variance)
        self._weights_variance = refresh_node(self._weights_variance)

        self._inputs = to_variable(inputs, requires_grad=True)
        mu = torch.mm(self._inputs, self._weights.unsqueeze(0).transpose(0, 1)).squeeze()

        reparameterized_normal = torch.distributions.normal.Normal(mu, self._noise_variance.expand(len(mu)))
        self._outputs = reparameterized_normal.rsample()
        self._outputs.reqiures_grad = True

        return CallResult(self._outputs.data.numpy(), has_finished = self._has_finished, iterations_done=self._iterations_done)


    def fit(self, *, timeout: float = None, iterations: int = None, fit_threshold: float, batch_size: int) -> CallResult:
        """
        Runs gradient descent for ``timeout`` seconds or ``iterations``
        iterations, whichever comes sooner, on log normal_density(self.weights * self.input
        - output, identity*self.noise_variance) +
        parameter_prior_primitives["weights"].score(self.weights) +
        parameter_prior_primitives["noise_variance"].score(noise_variance).
        """

        if self._new_training_data:
            self._weights = torch.FloatTensor(np.random.randn(self._training_inputs.size()[1]) * 0.001)
            self._noise_variance = torch.ones(1)
            # this should be a matrix
            self._weights_variance = torch.ones(1)
            self._new_training_data = False
        elif self._has_finished:
            return CallResult(None, has_finished=self._has_finished, iterations_done=self._iterations_done)

        if self._use_analytic_form:
            self._analytic_fit(iterations=iterations)
        else:
            self._gradient_fit(timeout=timeout, iterations=iterations, batch_size=batch_size)
        return CallResult(None, has_finished=self._has_finished, iterations_done=self._iterations_done)


    def _analytic_fit(self, *, iterations):
        train_x = self._training_inputs.data.numpy()
        train_y = self._training_outputs.data.numpy()

        cov_dim = self._training_inputs.shape[1]
        inv_covar = np.zeros([cov_dim, cov_dim])

        if self._weights_prior is not None:
            # just the prior on weights minus offset
            inv_covar[:cov_dim - 1, :cov_dim - 1] = np.linalg.inv(self._weights_prior.get_params()['covariance'])

        # this expression is (X^T*X + Lambda*I)^-1*X^T*Y
        # i.e. it is the solution to the problem argmin_w(E_D(w) + E_w(w))
        # where the first E_D(w) is the ML objective (least squares mvn)
        # and the second term E_w(w) is the regularizer, in this case Lambda/2*w^T*w
        w_sigma = np.dot(np.transpose(train_x), train_x) + inv_covar * float(self._noise_variance.data.numpy()[0])
        w_mu = np.dot(np.dot(np.linalg.inv(w_sigma), np.transpose(train_x)), train_y)

        self._weights = torch.FloatTensor(w_mu.flatten())
        self._weights_variance = torch.FloatTensor(w_sigma)

        self._iterations_done = 1
        self._has_finished    = True


    def _gradient_fit(self, *, timeout: float = None, iterations: int = 100, fit_threshold: float = 0, batch_size: int) -> None:
        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        if timeout is None:
            # TODO implement this
            timeout = np.inf

        if batch_size is None:
            # TODO come up with something more reasonable here
            batch_size = 1
        x_batches = []
        y_batches = []
        # otpionally do sampling with replacement
        for i in range(0, len(self._training_inputs), batch_size):
            x_batches.append(self._training_inputs[i:i+batch_size])
            y_batches.append(self._training_outputs[i:i+batch_size])
        num_batches = len(x_batches)

        start = time.time()
        # TODO fix this below to be the matrix again
        #  self._weights_variance = torch.Tensor()

        iter_count = 0
        has_converged = False
        while iter_count < iterations and has_converged == False:
            iter_count += 1
            batch_no = iter_count % num_batches

            grads = [self._gradient_params_log_likelihood(input=training_input,
                                                          output=training_output)
                     for training_input, training_output
                     in zip(x_batches[batch_no], y_batches[batch_no])]
            weights_grad = sum(grad[0] for grad in grads) * num_batches
            noise_grad = sum(grad[1] for grad in grads) * num_batches
            if self._weights_prior is not None:
                # TODO scale this by bz over total data
                weights_grad += torch.from_numpy(
                                    self._weights_prior.gradient_output(
                                        outputs=np.array([self._weights.data.numpy()]),
                                        inputs=[])
                                )
            self._weights.data += self._learning_rate * weights_grad * 1 / torch.norm(weights_grad)
            self._noise_variance.data += self._learning_rate * noise_grad * 1 / torch.norm(noise_grad)


            train_outputs = torch.mm(self._training_inputs, self._weights.unsqueeze(0).transpose(0, 1)).squeeze()
            train_y = self._training_outputs.data.numpy().flatten()
            rmse = mean_squared_error(train_outputs.data.numpy(), train_y)

            if rmse < self._best_rmse:
                self._converged_count = 0
                self._best_rmse = rmse
            else:
                self._converged_count += 1
            if self._converged_count > 1000:
                self._has_finished = True
                break

        self._iterations_done += iter_count


    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[ndarray]:
        """
        input : D-length numpy ndarray
        output : float
        Calculates
        log(normal_density(self.weights * self.input - output, identity * self.noise_variance))
        for a single input/output pair.
        """
        result = np.array([self._log_likelihood(output=to_variable(output),
                                              input=to_variable(input)).data.numpy()
                            for input, output in zip(inputs, outputs)])
        return CallResult(result)

    def log_likelihood(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[float]:

        inputs = self._offset_input(inputs=inputs)
        result = self.log_likelihoods(outputs=outputs, inputs=inputs, timeout=timeout, iterations=iterations)

        return CallResult(sum(result.value), has_finished=result.has_finished, iterations_done=result.iterations_done)

    def _log_likelihood(self, output: torch.autograd.Variable, input: torch.autograd.Variable) -> torch.autograd.Variable:
        """
        All inputs are torch tensors (or variables if grad desired).
        input : D-length torch to_variable
        output : float
        """
        expected_output = torch.dot(self._weights, input).unsqueeze(0)
        covariance = to_variable(self._noise_variance).view(1, 1)

        return log_mvn_likelihood(expected_output, covariance, output)

    def _gradient_params_log_likelihood(self, *, output: torch.autograd.Variable, input: torch.autograd.Variable) -> Tuple[torch.autograd.Variable, torch.autograd.Variable, torch.autograd.Variable]:
        """
        Output is ( D-length torch variable, 1-length torch variable )
        """

        self._weights = refresh_node(self._weights)
        self._noise_variance = refresh_node(self._noise_variance)
        log_likelihood = self._log_likelihood(output=output, input=input)
        log_likelihood.backward()
        return (self._weights.grad.data, self._noise_variance.grad.data)

    def _gradient_output_log_likelihood(self, *, output: ndarray, input: torch.autograd.Variable) -> torch.autograd.Variable:
        """
        output is D-length torch variable
        """

        output_var = to_variable(output)
        log_likelihood = self._log_likelihood(output=output_var, input=input)
        log_likelihood.backward()
        return output_var.grad

    def gradient_output(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Outputs]: # type: ignore
        """
        Calculates grad_output log normal_density(self.weights * self.input - output, identity * self.noise_variance)
        for a single input/output pair.
        """
        inputs = self._offset_input(inputs=inputs)

        outputs_vars = [to_variable(output, requires_grad=True) for output in outputs]
        inputs_vars = [to_variable(input) for input in inputs]
        grad = sum(self._gradient_output_log_likelihood(output=output,
                                                        input=input)
                   for (input, output) in zip(inputs_vars, outputs_vars))


        return grad.data.numpy()

    def gradient_params(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Params]:  # type: ignore
        """
        Calculates grad_weights fit_term_temperature *
        log normal_density(self.weights * self.input - output, identity * self.noise_variance)
        for a single input/output pair.
        """

        outputs_vars = [to_variable(output, requires_grad=True) for output in outputs]
        inputs_vars = [to_variable(input) for input in inputs]

        grads = [self._gradient_params_log_likelihood(output=output, input=input)
                 for (input, output) in zip(inputs_vars, outputs_vars)]
        grad_weights = sum(grad[0] for grad in grads)
        grad_noise_variance = sum(grad[1] for grad in grads)

        return Params(weights=grad_weights, offset=grad_offset, noise_variance=grad_noise_variance)


    def backward(self, *, gradient_outputs: Gradients[Outputs], fine_tune: bool = False, fine_tune_learning_rate: float = 0.00001) -> Tuple[Gradients[Inputs], Gradients[Params]]:  # type: ignore
        if self._inputs is None:
            raise Exception('Cannot call backpropagation before forward propagation. Call "produce" before "backprop".')
        else:
            if self._inputs.grad is not None:
                self._inputs.grad.data.zero_()

            self._outputs.backward(gradient=torch.Tensor(gradient_outputs))

            # this is the gradients given by end to end loss
            weights_grad = self._weights.grad.data
            noise_grad   = self._noise_variance.grad.data

            if fine_tune:
                # this is gradients given by the annealed local loss
                if self._fit_term_temperature != 0:
                    # TODO use minibatches here
                    training_grads = [self._gradient_params_log_likelihood(output=output, input=input)
                                      for (input, output) in zip(self._training_inputs, self._training_outputs)]
                    weights_grad += self._fit_term_temperature * \
                        sum(grad[0] for grad in training_grads)
                    noise_grad += self._fit_term_temperature * \
                        sum(grad[1] for grad in training_grads)

                # make local update with temperature if required
                # TODO add the score frmo the prior primitive here
                self._weights.data += weights_grad * 1 / torch.norm(weights_grad)
                self._noise_variance.data += noise_grad * 1 / torch.norm(noise_grad)

                self._weights = refresh_node(self._weights)
                self._noise_variance = refresh_node(self._noise_variance)

            grad_inputs = self._inputs.grad
            grad_params = Params(weights=ndarray(weights_grad[:-1]),
                                offset=float(weights_grad[-1]),
                                noise_variance=float(noise_grad[0]),
                                weights_variance=ndarray(np.zeros(self._weights_variance.shape)))

            if self._tune_prior_end_to_end:
                # update priors parameters here if specified
                self._weights_prior.backward(gradient_outputs=grad['weights'], fine_tune=True)

            return grad_inputs, grad_params

    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        self._fit_term_temperature = temperature

    def _offset_input(self, *, inputs: Inputs) -> Inputs:
        if inputs.shape[1] == self._weights.shape[0]:
            return inputs
        else:
            return np.insert(inputs, inputs.shape[1], 1, axis=1)

    def get_call_metadata(self) -> CallResult:
        return CallResult(None, has_finished=self._has_finished, iterations_done=self._iterations_done)

    def get_params(self) -> Params:
        return Params(
                    weights=ndarray(self._weights[:-1].data.numpy()),
                    offset=float(self._weights[-1].data.numpy()),
                    noise_variance=float(self._noise_variance.data.numpy()[0]),
                    weights_variance=ndarray(self._weights_variance.data.numpy())
               )

    def set_params(self, *, params: Params) -> None:
        full_weights = np.append(params['weights'], params['offset'])
        self._weights = to_variable(full_weights, requires_grad=True)
        self._weights.retain_grad()

        self._weights_variance = to_variable(params['weights_variance'], requires_grad=True)
        self._noise_variance = to_variable(params['noise_variance'], requires_grad=True)
