from d3m import container
from d3m.container import pandas # type: ignore
from d3m.container import ndarray
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.base import utils as base_utils

from d3m import utils as d3m_utils

# Import config file
from primitives_ubc.config_files import config

# Import relevant libraries
import os
import time
import logging
import numpy as np
import pandas as pd
import torch  # type: ignore
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple


__all__ = ('PrincipalComponentAnalysisPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

DEBUG = False  # type: ignore

class Params(params.Params):
    transformation: ndarray
    n_components: int


class Hyperparams(hyperparams.Hyperparams):
    max_components = hyperparams.Hyperparameter[int](
        default=0,
        description="max number of principled componenets to fit with",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    proportion_variance = hyperparams.Hyperparameter[float](
        default=1.0,
        description="regularization argument",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class PrincipalComponentAnalysisPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    PCA primitive convert a set of observations of possibly correlated variables into a set
    of values of linearly uncorrelated variables called principal components.
    It uses Singular Value Decomposition of the data to project it to a lower dimensional space.
    -------------
    Inputs:  DataFrame of features/inputs of shape: NxM, where N = samples and M = features/numerical (Attribute) inputs.
             or Denormalized DataFrame of dataset such as image dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "f44b6fd4-fd56-48d9-8fdc-c38300c8d256",
        "version": config.VERSION,
        "name": "Principal Component Analysis",
        "description": "Projects data to a lower dimensional space.",
        "python_path": "d3m.primitives.dimensionality_reduction.principal_component_analysis.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.DIMENSIONALITY_REDUCTION,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['pca', 'principal components'],
        "installation": [config.INSTALLATION],
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.hyperparams   = hyperparams
        self._random_state = random_seed
        self._verbose      = _verbose
        self._training_inputs:  Inputs  = None
        self._training_outputs: Outputs = None
        # Use GPU if available
        self._max_components = hyperparams['max_components']
        self._proportion_variance = hyperparams['proportion_variance']
        self._n_components   = None
        self._transformation = None
        self._mean           = None

    def _curate_data(self, training_inputs):
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
                new_XTrain.append(XTrain[arr].ravel())

            new_XTrain = np.array(new_XTrain)

        return new_XTrain, feature_columns_1


    def _remove_mean(self, data):
        """
        Takes a torch tensor, calculates the mean of each column and subtracts from it.
        -------------
        Input:  NxD torch array
        Output: D-length mean vector, NxD torch array
        -------------
        """
        N, D = data.size()
        mean = torch.zeros([D]).type(torch.DoubleTensor)
        for row in data:
            mean += row.view(D)/N
        zero_mean_data = data - mean.view(1, D).expand(N, D)

        return mean, zero_mean_data


    def _transform(self, training_inputs):
        # If already fitted with current training data, this call is a noop.
        if training_inputs is None:
            raise ValueError("Missing training data.")

        # Curate and get data from inputs
        XTrain, _ = self._curate_data(training_inputs=training_inputs)

        # Convert data from numpy array to torch tensor
        XTrain = torch.from_numpy(XTrain).type(torch.DoubleTensor)

        # Get eigenvectors of covariance
        self._mean, zero_mean_data = self._remove_mean(data=XTrain)
        cov  = torch.from_numpy(np.cov(XTrain.numpy().T))
        cov  = cov.type(torch.FloatTensor)
        e, V = torch.eig(cov, True)

        # Choose which/how many eigenvectors to use
        total_variance = sum(np.linalg.norm(e.numpy()[i, :])
                                for i in range(e.size()[0]))
        indices = []
        self._n_components = 0
        recovered_variance = 0
        while recovered_variance < self._proportion_variance * total_variance \
                and (self._max_components == 0 or self._n_components < self._max_components):
            best_index = max(range(e.size()[0]), key=lambda x: np.linalg.norm(e.numpy()[x, :]))
            indices.append(best_index)
            recovered_variance += np.linalg.norm(e.numpy()[best_index, :])
            e[best_index, :]    = torch.zeros(2)
            self._n_components += 1


        # Construct transformation matrix with eigenvectors
        self._transformation = torch.zeros([self._n_components, XTrain.size()[1]]).type(torch.DoubleTensor)
        for n, index in enumerate(indices):
            self._transformation[n, :] = V[:, index]

        self._fitted = True


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        """
        Inputs:  DataFrame of features
        Returns: Pandas DataFrame of the latent matrix
        """
        self._transform(training_inputs=inputs)

        # Curate data
        XTest, feature_columns = self._curate_data(training_inputs=inputs)

        # Convert to torch tensor
        XTest = torch.from_numpy(np.array(XTest)).type(torch.DoubleTensor)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(feature_columns)

        # Compute PCA
        pca_out = np.array([torch.mv(self._transformation, row - self._mean).numpy() for row in XTest])

        # Convert from ndarray from DataFrame
        pca_out = container.DataFrame(pca_out, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(pca_out.shape[1]):
            col_dict = dict(pca_out.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = "vector_" + str(col)
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
            pca_out.metadata            = pca_out.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

        # Append to outputs
        outputs = outputs.append_columns(pca_out)

        return base.CallResult(outputs)
