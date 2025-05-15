import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from mvu.dataset.meta import DatasetMeta, ImageDatasetMeta
from mvu.dataset.torch_utils import TorchDatasetSplits


def splitTrainingValidation(meta: DatasetMeta, trainingValidation: Dataset, testing: Dataset, validation_percent: float = 0.3, samples: Dict[str,int] = None) -> TorchDatasetSplits:
    """
    Shared code between CIFAR10 and StarCraft for splitting an image dataset
    :param meta:                Dataset meta for the result
    :param trainingValidation:  Dataset to split between training and validation
    :param testing:             Dataset to use for testing
    :param validation_percent:  Percentage of training data to use for validation
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :return:  Dataset instance
    """
    trainingValidationSize = len(trainingValidation)
    validationEnd = int(trainingValidationSize * validation_percent)
    trainingStart = validationEnd
    trainingEnd = trainingValidationSize
    if samples is not None:
        # training has a start offset and an end, so need some math to convert the limit
        if "train" in samples:
            trainLimit = samples["test"]
            if trainLimit < trainingEnd - trainingStart:
                trainingEnd = trainingStart + trainLimit
        # validate is easy to limit, just reduce max samples
        if "validate" in samples:
            validationEnd = min(validationEnd, samples["validate"])
        # only make test a subset if needed, can use the raw dataset otherwise
        if "test" in samples:
            testLimit = samples["test"]
            if testLimit < len(testing):
                testing = Subset(testing, range(0, testLimit))
    # apply the computed limits
    training = Subset(trainingValidation, range(trainingStart, trainingEnd))
    validation = Subset(trainingValidation, range(0, validationEnd))

    logging.info(f"Split training from {trainingValidationSize} into {len(training)} training images and {len(validation)} validation images. Found {len(testing)} testing images.")

    return TorchDatasetSplits(training, validation, testing, meta)


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


def _getCIFAR10Dataset(path: str, train: bool, transform: callable) -> Dataset:
    """Helper to ensure same parameters for train vs test in CIFAR10"""
    return CIFAR10(
        root=path,
        train=train,
        download=True,
        transform=transform
    )

def createCIFAR10Dataset(path: str = None, validation_percent: float = 0.3, samples: Dict[str,int] = None, zero_mean: bool = False, normalization: str = "cifar10", image_size: int = 32, sensor_size: int = 1) -> TorchDatasetSplits:
    """
    Helper to load in CIFAR10 in the format we expect.
    :param path:                Location to load the starcraft dataset into.
    :param validation_percent:  Percentage of training data to use for validation.
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :param zero_mean:           If true, transforms images to the range [-1, 1]. If false, leaves them at [0, 1]
    :param normalization:       Normalization method to use for images. Defaults to standard for CIFAR10
    :param sensor_size:         Size of sensors for making values missing.
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the CIFAR10 dataset"
    # insure the path exists, just to be safe
    Path(path).mkdir(exist_ok=True, parents=True)

    # we just know the classes directly, not sure if this can be fetched
    targets = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    logging.info(f"Using {len(targets)} CIFAR10 targets: {targets}")

    # create metadata
    meta = ImageDatasetMeta("cifar10", targets, image_size, sensor_size, 3)

    # fetch CIFAR10
    transformFunc = getTransform(image_size, 32, zero_mean, normalization)
    trainingValidation = _getCIFAR10Dataset(path, True, transformFunc)
    testing = _getCIFAR10Dataset(path, False, transformFunc)

    return splitTrainingValidation(meta, trainingValidation, testing, validation_percent=validation_percent, samples=samples)
