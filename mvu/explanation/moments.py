from typing import Tuple, Union, List

from torch import Tensor
from torch.distributions import Distribution, Beta

from .delta_dist import DeltaDistribution


def estimateBetaParametersFromMoments(mean: Tensor, var: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes parameters for a beta distribution using the mean and variance via method of moments.
    :param mean:  Sample mean
    :param var:   Sample variance
    :return:  Beta distribution parameters alpha then beta
    """
    assert mean.shape == var.shape, "Mean and variance must be the same shape"
    assert len(mean.shape) <= 1, "Mean must be a scalar or a vector"
    assert len(var.shape) <= 1, "Variance must be a scalar or a vector"

    upper = mean * (1 - mean)
    common = (upper / var) - 1
    swapped = (var >= upper).float()
    # alpha = mean * common
    # beta = (1 - mean) * common
    sign = 1 - (swapped * 2)
    # print(swapped)
    return (swapped + sign * mean) * common, (1 - swapped - sign * mean) * common
    # return (alpha * (1 - swapped) + beta * swapped,
    #         beta * (1 - swapped) + alpha * swapped)


def _toDistribution(mean, var, alpha, beta) -> Distribution:
    if var <= 0:
        return DeltaDistribution(mean)
    return Beta(alpha, beta)


def estimateDistributionFromMoments(mean: Tensor, var: Tensor) -> Union[Distribution, List[Distribution]]:
    alpha, beta = estimateBetaParametersFromMoments(mean, var)

    if len(mean.shape) == 0:
        return _toDistribution(mean, var, alpha, beta)
    return [_toDistribution(*params) for params in zip(mean, var, alpha, beta)]
