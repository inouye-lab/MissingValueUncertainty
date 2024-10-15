from typing import Tuple, Union, List

import torch
from torch import Tensor
from torch.distributions import Distribution, Beta

from .delta_dist import DeltaDistribution


def estimateBetaParametersFromMoments(mean: Tensor, var: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes parameters for a beta distribution using the mean and variance via method of moments.
    :param mean:  Mean tensor.
    :param var:   Variance tensor, must be broadcastable to the same size as the mean.
    :return:  Beta distribution parameters alpha then beta.
    """

    upper = mean * (1 - mean)
    common = (upper / var) - 1
    # indicates alpha and beta should be swapped for the given index
    swapped = torch.ge(var, upper).float()
    # the general formula for alpha is mean * common, while beta is (1-mean) * common
    # since we want to swap alpha and beta based on swapped, but also want to negate the two, it works out nicely
    # (mean-swapped) is either mean or (mean-1)=-(1-mean)
    # (1-mean-swapped) is either (1-mean) or -mean
    return (mean - swapped) * common, (1 - mean - swapped) * common


def _toDistribution(mean: Tensor, var: Tensor, alpha: Tensor, beta: Tensor) -> Distribution:
    """
    Helper to convert the given parameters to either a beta distribution or a delta distribution
    :param mean:    Scalar mean value
    :param var:     Scalar variance
    :param alpha:   Scalar alpha value. May be infinity if and only if variance is 0
    :param beta:    Scalar beta value. May be infinity if and only if variance is 0
    :return:  A delta distribution centered at the mean if the variance is 0, otherwise a beta distribution.
    """
    if var <= 0:
        return DeltaDistribution(mean)
    return Beta(alpha, beta)


def estimateBetaDistributionFromMoments(mean: Tensor, var: Tensor) -> Union[Distribution, List[Distribution]]:
    """
    Converts the given mean and variances into a distribution.
    Will map to a beta distribution if the variance is non-zero, or a delta distribution if zero.
    :param mean:  Mean tensor, can be scalar or a vector.
    :param var:   Variance tensor, must be the same size as mean.
    :return:  Single distribution if the input is scalar, or a list of distributions if the inputs are vectors
    """
    assert mean.shape == var.shape, "Mean and variance must be the same shape"
    assert len(mean.shape) <= 1, "Mean must be a scalar or a vector"
    alpha, beta = estimateBetaParametersFromMoments(mean, var)

    if len(mean.shape) == 0:
        return _toDistribution(mean, var, alpha, beta)
    return [_toDistribution(*params) for params in zip(mean, var, alpha, beta)]
