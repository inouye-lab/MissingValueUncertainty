from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from .dataset import Dataset
from .regressor import Regressor


def estimateResidual(regressor: Regressor, features: Union[Tensor, Dataset], targets: Tensor = None) -> Tensor:
    """
    Estimates the residual uncertainty for the given regressor and features tensor
    :param regressor: Regressor instance
    :param features:  Feature tensor of size `(samples, features)` with no missingness
    :param targets:   Targets for each sample, size `(samples,)`
    :return: Residual uncertainty for the whole model as a tensor of size 1
    """
    # allow passing a dataset as the second parameter instead of splitting it
    if targets is None:
        targets = features.targets
        features = features.features

    # make prediction, then return mean squared error
    prediction = regressor.predict(features)
    return torch.mean((targets - prediction) ** 2)


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
