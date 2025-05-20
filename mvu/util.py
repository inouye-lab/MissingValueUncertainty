import json
import logging
from json import JSONDecodeError
from typing import Union, Dict, List, Optional

import torch
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, StepLR
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


def jsonOrName(value: str) -> Dict:
    """
    Parses the value as JSON, if failing returns it as a dict with a key "name".
    Used to avoid the need to double quote string fallbacks.
    :param value:  Value to parse
    :return:   Json value parsed, falling back to the raw string value.
    """
    try:
        return json.loads(value)
    except JSONDecodeError:
        return {"name": str(value)}


def selectDevice(cuda_index: int) -> device:
    """
    Selects the device using the given index
    :param cuda_index:  Device index for GPU
    :return:
    """
    if cuda_index >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", index=cuda_index)
        logging.info(f"Using {device} for tensor calculations")
        # PyTorch lazy loads some of its modules which causes issues when in both GPU and threading if we happen to
        # try and load it on multiple threads at the same time. Workaround by using it before we dispatch.
        # see https://github.com/pytorch/pytorch/issues/90613 for more info
        torch.inverse(torch.ones((1, 1), device=device))
    else:
        device = torch.device("cpu")
        # we log whether CUDA is available to make it more clear if it was not an option or force disabled
        logging.info(f"Using {device} for tensor calculations, cuda available: {torch.cuda.is_available()}")

    return device


def getOptimizer(model: Module, name: str = "", **kwargs) -> Optimizer:
    """Creates an optimizer from the given arguments"""
    if name == "adam":
        logging.info(f"Adam optimizer configuration: {kwargs}")
        return Adam(model.parameters(), **kwargs)
    elif name == "sgd":
        logging.info(f"SGD optimizer configuration: {kwargs}")
        return SGD(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def getScheduler(optimizer: Optimizer, name: str = "none", **kwargs) -> Optional[LRScheduler]:
    """Creates an optimizer from the given arguments"""
    if name == "none":
        return None
    elif name == "step-decay":
        logging.info(f"Step-decay optimizer configuration: {kwargs}")
        return StepLR(optimizer, **kwargs)
    elif name == "cosine-annealing":
        logging.info(f"Cosine-annealing optimizer configuration: {kwargs}")
        return CosineAnnealingLR(optimizer, **kwargs)