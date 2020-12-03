import numpy as np
import numpy.matlib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from primitives_ubc.clfyCCFS.src.utils.ccfUtils import mat_unique
from primitives_ubc.clfyCCFS.src.utils.commonUtils import sVT
from primitives_ubc.clfyCCFS.src.utils.commonUtils import islogical

import logging
logger  = logging.getLogger(__name__)

def classExpansion(Y, N, optionsFor):
    """
    This function ensures that class data is in its binary expansion format

    Parameters
    ----------
    Y : pandas DataFrame/Numpy array
        Numpy array => For Numeric only
        Pandas DataFrame => For Numeric/String/Categorical
        Class information, can be a binary expansion, a numerical
        vector of labels or a cell array of numerical or string
        labels.
    N : Float
        Number of datapoints.
    optionsFor: dict
        Forest options

    Returns
    -------
    Y:  Numpy array
        Y in binary expansion format
    classes:  float
        Names of classes.  In CCT only the class index is stored and
        so this is used to convert to the original name.
    optionsFor: dict
        Updated forest options, e.g. because bSepPred has been
        switched on because non-mutually exclusive classes.
    """
    if isinstance(Y, pd.DataFrame):
        assert (Y.shape[1]==1), 'If Y is a DataFrame it should either be Nx1 for a single output'
        assert (not optionsFor["bSepPred"]), 'Seperate in-out prediction is only valid when Y is a logical array'
        # Encoding function
        enc = OneHotEncoder(handle_unknown='ignore')
        # Fit the classes
        enc.fit(Y)
        # Transform labels to one-hot categorical encoding
        classes = enc
        nCats   = len(enc.categories_[0])
        Y       = (enc.transform(Y)).toarray()
        # Setup task-ids
        optionsFor["task_ids"] = np.array([0])

    elif Y.shape[0] == N and Y.shape[1] == 1:
        assert (not optionsFor["bSepPred"]), 'Seperate in-out prediction is only valid when Y is a logical array'
        classes, _, Yindexes = mat_unique(Y)
        Y  = np.empty((Yindexes.shape[0], classes.size))
        Y.fill(False)
        for k in range(classes.size):
            Y[:, k] = (k == Yindexes)
        # Setup task-ids
        optionsFor["task_ids"] = np.array([0])

    elif islogical(Y) or (np.max(Y.flatten(order='F')) == 1 and np.min(Y.flatten(order='F')) == 0):
        N_c_present = np.cumsum(Y, axis=1)
        if np.all(N_c_present[:, -1] == 1) and (not optionsFor["bSepPred"]):
            optionsFor["task_ids"] = np.array([0])
            classes = np.arange(0, Y.shape[1])
        else:
            if (not optionsFor["bSepPred"]):
                optionsFor["bSepPred"] = True
                logger.warning('Providing a logical array with varying number of active classes, setting bSepPred to true.\
                                For multi-output classification use array of class indices where each column is an output')

            optionsFor["task_ids"] = np.arange(0, Y.shape[1])
            classes = np.matlib.repmat([False, True], 1, Y.shape[1])

    else:
        assert (not optionsFor["bSepPred"]), 'Seperate in-out prediction is only valid when Y is a logical array!'
        classes = {}
        Ycell   = {}

    if not isinstance(classes, type(OneHotEncoder(handle_unknown='ignore'))):
        if classes.shape[0] > (N-2):
            assert (False), ('More than n_data_points-2 classes appear to be present.\
                              Make sure no datapoints with missing output or regression option on!')


    return Y, classes, optionsFor
