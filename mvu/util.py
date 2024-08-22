import json
from json import JSONDecodeError
from typing import Union, Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .model.regressor import Regressor


@torch.no_grad()
def estimateResidual(regressor: Regressor, data: DataLoader, device: torch.device = None) -> Tensor:
    """
    Estimates the residual uncertainty for the given regressor and data loader
    :param regressor: Regressor instance
    :param data:      Data loader providing tuples of `(features, targets)` of sizes `(samples, features)`
                      and `(samples,)`, with no missingness
    :param device:    Device to use for tensor calculations. Returned result will also be on that device.
                      Expected to match the regressor's device
    :return: Residual uncertainty for the whole model as a tensor of size 1 on the passed device
    """
    squaredError = torch.tensor([0], device=device, dtype=torch.float)
    seenSamples = 0

    # simply process each batch one at a time, no need to do anything fancy with loaders
    for (features, targets) in data:
        if device is not None:
            features = features.to(device)
            targets = targets.to(device)
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


def jsonOrString(value: str) -> Union[int, str, Dict, List]:
    """
    Parses the value as JSON, if failing returns it as a raw string.
    Used to avoid the need to double quote string fallbacks.
    :param value:  Value to parse
    :return:   Json value parsed, falling back to the raw string value.
    """
    try:
        return json.loads(value)
    except JSONDecodeError:
        return str(value)