import numpy as np
from primitives.clfyCCFS.src.utils.commonUtils import is_numeric
from primitives.clfyCCFS.src.utils.commonUtils import islogical

def treeOutputsToForestPredicts(CCF, treeOutputs):
    """
    Converts outputs from individual trees to forest predctions and
    probabilities.

    Parameters
    ----------
    CCF = Output of genCCF
    treeOutputs = Array typically generated by predictCCF. Description provided
                  in doc string of predictCCF as it is provided as an output.
    """
    forestProbs = np.squeeze(np.mean(treeOutputs, axis=1))

    if CCF["options"]["bSepPred"]:
        forestPredicts = forestProbs > 0.5
    else:
        # Check if task_ids is single number
        if type(CCF["options"]["task_ids"]) == int:
            if CCF["options"]["task_ids"] == 1:
                task_ids_size  = 1
                forestPredicts = np.empty((forestProbs.shape[0], task_ids_size))
                forestPredicts.fill(np.nan)
                forestPredicts[:, 0] = np.argmax(forestProbs, axis=1)
            else:
                task_ids_size  = 1
                forestPredicts = np.empty((forestProbs.shape[0], task_ids_size))
                forestPredicts.fill(np.nan)
                for nO in range((task_ids_size)-1):
                    forestPredicts[:, nO] = np.argmax(forestProbs[:, CCF["options"]["task_ids"]:(CCF["options"]["task_ids"]+1)-1], axis=1)
                forestPredicts[:, -1] = np.argmax(forestProbs[:, CCF["options"]["task_ids"]:], axis=1)
        else:
            forestPredicts = np.empty((forestProbs.shape[0], CCF["options"]["task_ids"].size))
            forestPredicts.fill(np.nan)
            for nO in range((CCF["options"]["task_ids"].size)-1):
                forestPredicts[:, nO] = np.argmax(forestProbs[:, CCF["options"]["task_ids"][nO]:(CCF["options"]["task_ids"][nO+1]-1)], axis=1)
            forestPredicts[:, -1] = np.argmax(forestProbs[:, CCF["options"]["task_ids"][-1]:], axis=1)
        # Convert to type int
        forestPredicts = forestPredicts.astype(int)

    if is_numeric(CCF["classNames"]):
        if islogical(forestPredicts):
            assert (forestPredicts.shape[1] == 1), 'Class names should have been a cell if multiple outputs!'
            #print(forestPredicts)
            forestPredicts = CCF["classNames"][forestPredicts+1]
        else:
            forestPredicts = CCF["classNames"][forestPredicts]

    # Fix needed -- Support for DataFrame
    elif isinstance(CCF["classNames"], pd.DataFrame):
        assert (CCF["classNames"].size == forestPredicts.shape[1]), 'Number of predicts does not match the number of outputs in classNames'

    elif islogical(CCF["classNames"]) and CCF["classNames"].size:
        forestPredicts = (forestPredicts == 2)


    return forestPredicts, forestProbs
