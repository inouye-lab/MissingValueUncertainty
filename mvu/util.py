import torch
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader

from .model.regressor import Regressor


@torch.nograd
def estimateResidual(regressor: Regressor, data: DataLoader) -> Tensor:
    """
    Estimates the residual uncertainty for the given regressor and data loader
    :param regressor: Regressor instance
    :param data:      Data loader providing tuples of `(features, targets)` of sizes `(samples, features)`
                      and `(samples,)`, with no missingness
    :return: Residual uncertainty for the whole model as a tensor of size 1
    """
    squaredError = Tensor([0])
    seenSamples = 0

    # simply process each batch one at a time, no need to do anything fancy with loaders
    for (features, targets) in data:
        means = regressor.predict(features)
        squaredError += ((means - targets) ** 2).sum()
        seenSamples += targets.shape[0]
    assert seenSamples != 0, "No samples in empirical uncertainty method"
    return squaredError / seenSamples


def gaussianLogLikelihood(squaredError: Tensor, var: Tensor) -> Tensor:
    """
    Evaluates the log likelihood for a gaussian distribution
    :param squaredError:  Squared difference between true value and expected value
    :param var:           Predicted variance, should be same size as expected
    :return:  Log-likelihood score for each sample
    """
    clampVar = var.clamp(min=1e-10)
    return -0.5 * torch.log(torch.mul(2 * torch.pi, clampVar))\
        - 0.5 / clampVar * squaredError
