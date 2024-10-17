from typing import Tuple, Union, List

import torch
from overrides import override
from torch import Tensor, Generator
from torch.distributions import Distribution, Beta

from .decision import DecisionMaker, computeBestActions
from .delta_dist import DeltaDistribution
from ..model.method import Method


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


class MethodOfMomentsDecisionMaker(DecisionMaker):
    """
    Makes decisions using the method of moments to map a method to a distribution, then samples it to estimate actions.
    """

    method: Method
    """Method to run, producing a mean and variance"""
    size: torch.Size
    """Samples to take from the distribution"""
    momentMatcher: callable
    """
    Method matching a tensor of means and a tensor of variances to a distribution (for shape size 0)
    or a list of distributions (for shape size 1). See `estimateBetaDistributionFromMoments`.
    """

    def __init__(self, method: Method, distSamples: int, momentMatcher: callable = estimateBetaDistributionFromMoments):
        self.method = method
        self.size = torch.Size((distSamples,))
        self.momentMatcher = momentMatcher

    @override
    def estimateBestAction(self, features: Tensor, lossFunction: callable, actions: Tensor, rand: Generator = None,
                           indices: Tensor = None) -> Tuple[Tensor, Tensor]:
        inputSamples = features.shape[0]
        mean, var = self.method.predictWithUncertainty(features, rand=rand, indices=indices)
        assert mean.shape[0] == inputSamples
        assert var.shape[0] == inputSamples
        distributions: List[Distribution] = self.momentMatcher(mean, var)
        assert len(distributions) == inputSamples
        # TODO: not sure how to enforce the random state in torch distributions
        phis = [dist.sample(self.size) for dist in distributions]
        return computeBestActions(phis, lossFunction, actions)

    @override
    def supportsIndices(self, indices: Tensor) -> Tensor:
        return self.method.supportsIndices(indices)

