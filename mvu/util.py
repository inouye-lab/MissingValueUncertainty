import json
import logging
from json import JSONDecodeError
from typing import Union, Dict, List

import torch
from torch import Tensor, device
from torch.utils.data import DataLoader

from mvu.model.regressor import Regressor


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

def process_tensor(target_attr, seed_local=None):
    # Create a generator for local seed control
    rng = torch.Generator()

    if seed_local is not None:
        rng.manual_seed(seed_local)  # Set the seed for this generator only

    for i in range(target_attr.size(0)):  # Iterate over rows
        row = target_attr[i]
        ones_count = row.sum().item()  # Count the number of 1s in the row

        if ones_count > 1:
            # Find indices of 1s
            one_indices = (row == 1).nonzero(as_tuple=True)[0]
            # Randomly select one index to keep as 1, using the local generator
            keep_index = one_indices[torch.randint(0, int(ones_count), (1,), generator=rng).item()]
            # Set all other indices to 0
            row.fill_(0)
            row[keep_index] = 1
        elif ones_count == 0:
            # Randomly select one index to set as 1, using the local generator
            random_index = torch.randint(0, int(row.size(0)), (1,), generator=rng).item()
            row[random_index] = 1
        # If ones_count == 1, do nothing

    return target_attr