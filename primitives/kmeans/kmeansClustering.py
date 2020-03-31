from d3m import container
from d3m.container import pandas
from d3m.container.numpy import ndarray
from d3m.primitive_interfaces import base
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.base import utils as base_utils
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from d3m import utils as d3m_utils

# Import config file
from primitives.config_files import config

# Import relevant libraries
import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as KMeans_  # type: ignore
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple

__all__ = ('KMeansClusteringPrimitive',)
logger  = logging.getLogger(__name__)

Inputs  = container.DataFrame
Outputs = container.DataFrame

DEBUG = False  # type: ignore

class Params(params.Params):
    cluster_centers: ndarray  # Coordinates of cluster centers.


class Hyperparams(hyperparams.Hyperparams):
    """
    Hyper-parameters for this primitive.
    """
    n_clusters = hyperparams.Hyperparameter[int](
        default=8,
        description="The number of clusters to form as well as the number of centroids to generate.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    init = hyperparams.Hyperparameter[Union[str, ndarray]](
        default='k-means++',
        description='{‘k-means++’, ‘random’} or ndarray of shape (n_clusters, n_features). If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    n_init = hyperparams.Hyperparameter[int](
        default=10,
        description='Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    max_iter = hyperparams.Hyperparameter[int](
        default=300,
        description='Maximum number of iterations of the k-means algorithm for a single run.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    tol = hyperparams.Hyperparameter[float](
        default=0.0001,
        description='Relative tolerance with regards to inertia to declare convergence.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    n_jobs = hyperparams.Hyperparameter[int](
        default=1,
        description='The number of jobs to use for the computation.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    algorithm = hyperparams.Enumeration[str](
        values=['auto', 'full', 'elkan'],
        default='auto',
        description='The classical EM-style algorithm is “full”. The “elkan” variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. “auto” chooses “elkan” for dense data and “full” for sparse data.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class KMeansClusteringPrimitive(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
    minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below).
    This algorithm requires the number of clusters to be specified. It scales well to large number
    of samples and has been used across a large range of application areas in many different fields.
    Note: If the algorithm stops before fully converging (because of tol or max_iter),
    labels_ and cluster_centers_ will not be consistent, i.e. the cluster_centers_ will not be the means
    of the points in each cluster. Also, the estimator will reassign labels_ after the last iteration to
    make labels_ consistent with predict on the training set.
    -------------
    Inputs:  DataFrame of features/inputs of shape: NxM, where N = samples and M = features/numerical (Attribute) inputs.
    Outputs: DataFrame containing the target column of shape Nx1 or denormalized dataset.
    -------------
    """
    # Metadata
    __author__ = 'UBC DARPA D3M Team, Tony Joseph <tonyjos@cs.ubc.ca>'
    metadata   =  metadata_base.PrimitiveMetadata({
        "id": "6e8cf1c9-fcef-45d9-9e30-d984f7b4b561",
        "version": config.VERSION,
        "name": "KMeans Clustering",
        "description": "A unsupervised learning algorithm for clustering data.",
        "python_path": "d3m.primitives.clustering.kmeans.UBC",
        "primitive_family": metadata_base.PrimitiveFamily.CLUSTERING,
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.K_MEANS_CLUSTERING],
        "source": {
            "name": config.D3M_PERFORMER_TEAM,
            "contact": config.D3M_CONTACT,
            "uris": [config.REPOSITORY],
        },
        "keywords": ['k-means', 'unsupervised learning'],
        "installation": [config.INSTALLATION],
        "hyperparams_to_tune": ['n_init', 'max_iter']
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        try:
            self._kmeans = KMeans_(n_clusters=hyperparams['n_clusters'],
                                   init=hyperparams['init'],
                                   n_init=hyperparams['n_init'],
                                   max_iter=hyperparams['max_iter'],
                                   tol=hyperparams['tol'],
                                   verbose=_verbose,
                                   random_state=random_seed,
                                   n_jobs=hyperparams['n_jobs'],
                                   algorithm=hyperparams['algorithm'])
        except:
            # If passed in a initial hyperparam that was invalid, then default in most cases
            self._kmeans = KMeans_(n_clusters=hyperparams['n_clusters'],
                                   init='k-means++',
                                   n_init=hyperparams['n_init'],
                                   max_iter=hyperparams['max_iter'],
                                   tol=0.0001,
                                   verbose=0,
                                   random_state=random_seed,
                                   n_jobs=None,
                                   algorithm='auto')


        self._fitted = False



    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs   = inputs
        self._new_training_data = True


    def _curate_data(self, training_inputs, get_labels):
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
                XTrain_Flatten = (XTrain[arr]).flatten()
                new_XTrain.append(XTrain_Flatten)
            new_XTrain = np.array(new_XTrain)

        if get_labels:
            # Training labels
            YTrain = np.array([])

            # Get label column names
            label_name_columns  = []
            label_name_columns_ = list(training_inputs.columns)
            for lbl_c in label_columns:
                label_name_columns.append(label_name_columns_[lbl_c])

            # Get labelled dataset if available
            try:
                label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            except ValueError:
                label_columns  = training_inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
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

        return new_XTrain, feature_columns_1


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        # Curate data
        XTrain, _ = self._curate_data(training_inputs=self._training_inputs, get_labels=False)

        self._kmeans = self._kmeans.fit(XTrain)
        self._fitted = True

        return base.CallResult(None)


    def produce(self, *, inputs: Inputs, iterations: int = None, timeout: float = None) -> base.CallResult[Outputs]:
        """
        Inputs:  DataFrame of features or numerical inputs
        Returns: Pandas DataFrame Containing predictions
        """
        # Inference
        if not self._fitted:
            raise ValueError('Please fit the model before calling produce!')

        # Curate data, outputs given
        XTest, YTest, feature_columns, label_name_columns = self._curate_data(training_inputs=inputs, get_labels=True)

        add_class_index = False
        if YTest.size > 0:
            # Check if class index is from 0 to C-1 or 1 to C
            if (0.0 in YTest[:, 0]) or (0 in YTest[:, 0]):
              add_class_index = False
            else:
              add_class_index = True
        else:
            XTest, feature_columns = self._curate_data(training_inputs=inputs, get_labels=False)

        # Delete columns with path names of nested media files
        outputs = inputs.remove_columns(feature_columns)

        # Predictions
        predictions = self._kmeans.predict(XTest)
        if add_class_index:
            predictions = np.add(predictions, 1)

        # Convert from ndarray from DataFrame
        predictions = container.DataFrame(predictions, generate_metadata=True)

        # Update Metadata for each feature vector column
        for col in range(predictions.shape[1]):
            col_dict = dict(predictions.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name']            = label_name_columns[col]
            col_dict["semantic_types"]  = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/PredictedTarget",)
            predictions.metadata        = predictions.metadata.update((metadata_base.ALL_ELEMENTS, col), col_dict)

        if len(label_name_columns) != 0:
            # Rename Columns to match label columns
            predictions.columns = label_name_columns

        # Append predictions to outputs
        outputs = outputs.append_columns(predictions)

        return base.CallResult(outputs)


    def get_params(self) -> Params:
        return Params(cluster_centers=self._kmeans.cluster_centers)


    def set_params(self, *, params: Params) -> None:
        return None
