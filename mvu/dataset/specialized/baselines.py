import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

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

def _getCIFAR10Dataset(path: str, train: bool) -> Dataset:
    """Helper to ensure same parameters for train vs test in CIFAR10"""
    return CIFAR10(
        root=path,
        train=train,
        download=True,
        transform=lambda image: (to_tensor(image).to(torch.float32) * 2) - 1
    )

def createCIFAR10Dataset(path: str = None, validation_percent: float = 0.3, samples: Dict[str,int] = None, sensor_size: int = 1) -> TorchDatasetSplits:
    """
    Helper to load in CIFAR10 in the format we expect.
    :param path:                Location to load the starcraft dataset into.
    :param validation_percent:  Percentage of training data to use for validation.
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
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
    meta = ImageDatasetMeta("cifar10", targets, 32, sensor_size, 3)

    # fetch CIFAR10
    trainingValidation = _getCIFAR10Dataset(path, True)
    testing = _getCIFAR10Dataset(path, False)

    return splitTrainingValidation(meta, trainingValidation, testing, validation_percent=validation_percent, samples=samples)
