import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import CIFAR10

from mvu.dataset.meta import ImageDatasetMeta
from mvu.dataset.torch_utils import TorchDatasetSplits


def _transformZeroMean(tensor: Tensor):
    """Transformation used to match the CelebA diffusion model"""
    return (tensor.to(torch.float32) * 2) - 1

def getTransform(image_size: int, original_size: int, zero_mean: bool = False, normalization: str = "none") -> callable:
    """Gets the image transform for the given name and sizes."""

    toApply: List[callable] = []
    # resize if requested
    if image_size != original_size:
        logging.info(f"Resizing images from {original_size} to {image_size}")
        toApply.append(transforms.Resize(image_size))

    # always convert to tensor
    toApply.append(transforms.ToTensor())

    # compatability with CelebA diffusion model
    if zero_mean:
        logging.info(f"Applying simple zero mean to {image_size}")
        toApply.append(_transformZeroMean)

    # select normalization
    if normalization == "cifar10":
        logging.info(f"Applying CIFAR10 standard normalization")
        toApply.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]))
    elif normalization == "0.5":
        logging.info(f"Applying 0.5 normalization")
        toApply.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    elif normalization != "none":
        raise ValueError(f"Unknown transform: {normalization}")

    # return final combination
    return transforms.Compose(toApply)


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
"""Class list from CIFAR10, for the sake of our debug output"""

def createCIFAR10Dataset(path: str = None, validation_percent: float = 0.3, samples: Dict[str,int] = None, normalization: str = "cifar10", image_size: int = 32, sensor_size: int = 1, **kwargs) -> TorchDatasetSplits:
    """
    Helper to load in CIFAR10 in the format we expect.
    :param path:                Location to load the starcraft dataset into.
    :param validation_percent:  Percentage of training data to use for validation.
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :param normalization:       Normalization method to use for images. Defaults to standard for CIFAR10
    :param image_size:          Image size to use for images. Defaults to 32; if larger will rescale.
    :param sensor_size:         Size of sensors for making values missing.
    :param kwargs:              Additional image transform parameters
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the CIFAR10 dataset"
    # insure the path exists, just to be safe
    Path(path).mkdir(exist_ok=True, parents=True)

    # we just know the classes directly, not sure if this can be fetched
    logging.info(f"Using {len(CIFAR10_CLASSES)} CIFAR10 targets: {CIFAR10_CLASSES}")

    # create metadata
    meta = ImageDatasetMeta("cifar10", CIFAR10_CLASSES, image_size, sensor_size, 3)

    # fetch CIFAR10
    transformFunc = getTransform(image_size, 32, normalization=normalization, **kwargs)
    trainingValidation = CIFAR10(root=path, train=True,  download=True, transform=transformFunc)
    testing            = CIFAR10(root=path, train=False, download=True, transform=transformFunc)

    return TorchDatasetSplits.split(meta, trainingValidation, testing, validation_percent=validation_percent, samples=samples)
