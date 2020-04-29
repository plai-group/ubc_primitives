import torch  # type: ignore
import numpy as np  # type: ignore
from torch.autograd import Variable  # type: ignore

def to_variable(value: Any, requires_grad: bool = False) -> Variable:
    """
    Converts an input to torch Variable object
    input
    -----
    value - Type: scalar, Variable object, torch.Tensor, numpy ndarray
    requires_grad  - Type: bool . If true then we require the gradient of that object

    output
    ------
    torch.autograd.variable.Variable object
    """

    if isinstance(value, Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value.float(), requires_grad=requires_grad)
    elif isinstance(value, np.ndarray) or isinstance(value, ndarray):
        return Variable(torch.from_numpy(value.astype(float)).float(), requires_grad=requires_grad)
    elif value is None:
        return None
    else:
        return Variable(torch.Tensor([float(value)]), requires_grad=requires_grad)


def log_mvn_likelihood(mean: torch.FloatTensor, covariance: torch.FloatTensor, observation: torch.FloatTensor) -> torch.FloatTensor:
    """
    all torch primitives
    all non-diagonal elements of covariance matrix are assumed to be zero
    """
    k = mean.shape[0]
    variances = covariance.diag()
    log_likelihood = 0
    for i in range(k):
        log_likelihood += - 0.5 * torch.log(variances[i]) \
                          - 0.5 * k * math.log(2 * math.pi) \
                          - 0.5 * ((observation[i] - mean[i])**2 / variances[i])
    return log_likelihood


def refresh_node(node: Variable) -> Variable:
    return torch.autograd.Variable(node.data, True)
