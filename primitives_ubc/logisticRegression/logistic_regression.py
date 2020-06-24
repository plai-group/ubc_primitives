from d3m import container
from d3m.container.numpy import ndarray
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin
from d3m.primitive_interfaces.base import SamplingCompositionalityMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import Gradients
from d3m.primitive_interfaces.base import CallResult

# Import config file
from primitives_ubc.config_files import config

import os
import time
import math
import random
import importlib
import numpy  as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from typing import Any, cast, Dict, List, Union, Sequence, Optional, Tuple

__all__ = ('LogisticRegressionPrimitive',)

Inputs  = container.DataFrame
Outputs = container.DataFrame

class ImportModules:
    """
    Import heavy modules after calling the primitive as not to slow down d3m.index
    """
    _weight_files = []
    def __init__(self):
        self._initialized = False

    def _import_lib(self):
        if self._initialized:
            return

        global theano, pm, MultiTrace
        global Model, Normal, Bernoulli, NUTS
        global invlogit, sample, sample_ppc, find_MAP

        theano     = importlib.import_module('theano')
        pm         = importlib.import_module('pymc3')
        Model      = importlib.import_module('pymc3', 'Model')
        Normal     = importlib.import_module('pymc3', 'Normal')
        Bernoulli  = importlib.import_module('pymc3', 'Bernoulli')
        NUTS       = importlib.import_module('pymc3', 'NUTS')
        invlogit   = importlib.import_module('pymc3', 'invlogit')
        sample     = importlib.import_module('pymc3', 'sample')
        sample_ppc = importlib.import_module('pymc3', 'sample_ppc')
        find_MAP   = importlib.import_module('pymc3', 'find_MAP')
        MultiTrace = importlib.import_module('pymc3.backends.base', 'MultiTrace')

        self._initialized = True


class Params(params.Params):
    weights: Optional[Any]
    _categories: Optional[Any]
    target_names_: Optional[List[str]]


class Hyperparams(hyperparams.Hyperparams):
    burnin = hyperparams.Hyperparameter[int](
        default=1000,
        description='The number of samples to take before storing them',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    mu = hyperparams.Hyperparameter[float](
        default=0.0,
        description='Mean of Gaussian prior on weights',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    sd = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Standard deviation of Gaussian prior on weights',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    num_iterations = hyperparams.Hyperparameter[int](
        default=1000,
        description="Number of iterations to sample the model.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class LogisticRegressionPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                  SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                  SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                                  ImportModules):
    """
    -------------
    Inputs:  DataFrame of features of shape: NxM, where N = samples and M = features.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "8a992784-3b41-4915-bb47-3ff00e51e60f",
        "version": config.VERSION,
        "name": "Bayesian Logistic Regression",
        "description": "A bayesian approach to Logistic Regression",
        "python_path": "d3m.primitives.classification.logistic_regression.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOGISTIC_REGRESSION,],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['bayesian', 'binary classification'],
        "installation": [config.INSTALLATION],
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = random_seed
        self._verbose      = _verbose
        self._training_inputs: Inputs   = None
        self._training_outputs: Outputs = None

        # Intialize ImportModules
        ImportModules.__init__(self)
        # Import other needed modules
        ImportModules._import_lib(self)

        # Set parameters
        self._mu = self.hyperparams['mu']
        self._sd = self.hyperparams['sd']
        self._burnin = self.hyperparams['burnin']

        self._trace = None  # type: MultiTrace
        self._model = None  # type: Model

        # Is the model fit on data
        self._fitted = False


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
            new_XTrain = training_inputs.iloc[:, feature_columns_1]
            new_XTrain = self._to_numeric_and_fill_missing_vals(new_XTrain)
            new_XTrain = (new_XTrain.to_numpy()).astype(np.float)
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
            YTrain = (training_outputs.iloc[:, label_columns])
            YTrain, _categories = self._to_numeric_and_fill_missing_vals(YTrain, return_categories=True)
            YTrain = (YTrain.to_numpy()).astype(np.int)
            self._categories = _categories

            return new_XTrain, YTrain, feature_columns_1

        return new_XTrain, feature_columns_1


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        inputs, outputs, _ = self._curate_data(training_inputs=inputs, training_outputs=outputs, get_labels=True)
        self._training_inputs   = inputs
        self._training_outputs  = outputs
        self._new_training_data = True


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Returns the MAP estimate of p(y | x = inputs; w)
        inputs: (num_inputs,  D) numpy array
        outputs : numpy array of dimension (num_inputs)
        """
        if not self._fitted:
            raise Exception('Please fit the model before calling produce!')

        # Curate data
        XTest, feature_columns = self._curate_data(training_inputs=inputs, training_outputs=None, get_labels=False)

        w = self._trace['weights']
        w = np.squeeze(w, axis=2)

        predictions = (np.einsum('kj,ij->i', w, XTest) > 0).astype(int)
        predictions_binary = np.eye(2)[predictions]

        # Inverse map categories
        predictions = self._categories.inverse_transform(predictions_binary)

        # Convert from ndarray from DataFrame
        predictions = container.DataFrame(predictions, generate_metadata=True)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(feature_columns)

        # Update Metadata for each feature vector column
        for col in range(predictions.shape[1]):
            col_dict = dict(predictions.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = self.label_name_columns[col]
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
            predictions.metadata        = predictions.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)
        # Rename Columns to match label columns
        predictions.columns = self.label_name_columns

        # Append to outputs
        outputs = outputs.append_columns(predictions)

        return base.CallResult(outputs)


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult:
        """
        Sample a Bayesian Logistic Regression model using NUTS to find
        some reasonable weights.
        """
        if self._fitted:
            return base.CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        # make sure that all training outputs are either 0 or 1
        if not ((self._training_outputs == 0) | (self._training_outputs == 1)).all():
            raise ValueError("training outputs must be either 0 or 1")

        # Set all files
        if iterations == None:
            _iterations = self.hyperparams['num_iterations']
        else:
            _iterations = iterations

        # training data needs to be a Theano shared variable for
        # the later produce code to work
        _, n_features = self._training_inputs.shape
        self._training_inputs  = theano.shared(self._training_inputs)
        self._training_outputs = theano.shared(self._training_outputs)

        # As the model depends on number of features it has to be here
        # and not in __init__
        with Model.Model() as model:
            weights = Normal.Normal('weights', mu=self._mu, sd=self._sd, shape=(n_features, 1))
            p = invlogit.invlogit(pm.math.dot(self._training_inputs, weights))
            Bernoulli.Bernoulli('y', p, observed=self._training_outputs)
            trace = sample.sample(_iterations,
                                  random_seed=self.random_seed,
                                  trace=self._trace,
                                  tune=self._burnin, progressbar=False)
        self._trace  = trace
        self._model  = model
        self._fitted = True

        return CallResult(None)


    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> CallResult[Sequence[Outputs]]:
        # Set shared variables to test data, outputs just need to be the correct shape
        self._training_inputs.set_value(inputs)
        self._training_outputs.set_value(np.random.binomial(1, 0.5, inputs.shape[0]))

        with self._model:
            post_pred = sample_ppc.sample_ppc(self._trace,
                                              samples=num_samples,
                                              progressbar=False)

        return CallResult(post_pred['y'].astype(int))


    def _to_numeric_and_fill_missing_vals(self, dataframe, return_categories=False):
        # Missing values
        imp1 = SimpleImputer(missing_values='', strategy="most_frequent")
        imp2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        # Encoder
        enc  = OneHotEncoder(handle_unknown='ignore')

        i = 0
        all_types = []
        for col in dataframe.columns:
            try:
                dataframe[[col]] = imp1.fit_transform(dataframe[[col]])
                dataframe[[col]] = imp2.fit_transform(dataframe[[col]])
            except ValueError:
                # Assuimg empty column and replace nan with 0
                dataframe[col].fillna(0, inplace=True)
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except ValueError:
                # Assuming string
                dataframe[col] = np.argmax(enc.fit_transform(dataframe[[col]]).toarray(), axis=1)
            # Replace nan with 0
            dataframe[col].fillna(0, inplace=True)
            i += 1

        if return_categories:
            return dataframe, enc

        return dataframe


    def _log_likelihood(self, *, input: Inputs, output: Outputs) -> float:
        """
        Provides a likelihood of one output given the inputs and weights
        L_(y | x; w) = log(p(y | x; w)) = log(p) if y = 0 else log(1 - p)
            where: p = invl(w^T * x)
             invl(x) = exp(x) / 1 + exp(x)
        """
        logp = self._model.logp
        weights = self._trace["weights"]
        self._training_inputs.set_value(input)
        self._training_outputs.set_value(output)
        return float(np.array([logp(dict(y=output, weights=w)) for w in weights]).mean())


    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Sequence[float]]:
        """
        Provides a likelihood of the data given the weights
        """
        return CallResult(np.array([self._log_likelihood(input=[input], output=[output]) for input, output in zip(inputs, outputs)]))


    def get_params(self) -> Params:
        w = self._trace['weights']

        return Params(weights=w,\
                      _categories=self._categories,\
                      target_names_=self.label_name_columns)


    def set_params(self, *, params: Params) -> None:
        self._trace = {}
        self._trace['weights']  = params["weights"]
        self._categories        = params["_categories"]
        self.label_name_columns = params["target_names_"]
        self._fitted            = True


    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['random_state'] = self._random_state

        return state


    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']
