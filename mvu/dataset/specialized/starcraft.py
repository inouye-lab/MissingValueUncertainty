import logging
from pathlib import Path
from typing import Tuple, List, Union, Dict

import torch
from overrides import override
from sc2image import StarCraftImage, StarCraftCIFAR10
from torch import Tensor
from torch.utils.data import Dataset, Subset

from mvu.dataset.meta import ImageDatasetMeta
from mvu.dataset.torch import TorchDatasetSplits
from torchvision.transforms.functional import to_tensor


class StarCraftDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Wrapper around the starcraft image dataset class put the data in the format we expect"""

    base: StarCraftImage
    """Base dataset instance, will fetch data from it"""
    targets: List[str]
    """Metadata variable to use as regression target"""

    def __init__(self, path: str, imageFormat: str, imageSize: int, targets: List[str], train: bool):
        self.base = StarCraftImage(path, image_format=imageFormat, image_size=imageSize, train=train,
                                   return_dict=True, use_metadata_cache=True, download=True)
        self.targets = targets
        if len(self.targets) == 0:
            logging.warning("No target for StarCraft dataset, this will give unexpected behavior in regression tasks.")

    def __len__(self):
        return len(self.base)

    @override
    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        (unitIds, unitValues), data = self.base[item]
        # if no target is defined, just use 0. This just makes it simpler in contexts that are not regressing
        targets: Tensor
        if len(self.targets) == 0:
            targets = torch.tensor([0])
        else:
            metadata = data["metadata"]
            targets = torch.tensor([metadata[target] for target in self.targets])
        # expect vectors for the input instead of an image
        # we also convert to floats so we can actually place nan in the tensor for missingness
        # TODO: should we be converting to a range of -1 to 1 instead of 0 to 1?
        return (unitValues.float() / 255).reshape(-1), targets


def createDataset(path: str, train: bool, image_format: str = 'bag-of-units-first', image_size: int = 64,
                  targets: Union[str, List[str]] = None) -> Dataset:
    if image_format == "cifar10":
        assert image_size == 32, "StarCraft CIFAR10 requires image size of 32"
        assert targets is not None, "StartCraft CIFAR10 does not support targets"
        return StarCraftCIFAR10(
            root=path,
            train=train,
            download=True,
            transform=lambda image: (to_tensor(image).to(torch.float32) * 2) - 1
        )
    else:
        # TODO: can directly use the original class likely
        return StarCraftDataset(path, image_format, image_size, targets, train)


def createStarCraftDataset(path: str = None, targets: Union[str, List[str]] = None,
                           validation_percent: float = 0.3, samples: Dict[str,int] = None,
                           image_format: str = 'bag-of-units-first',
                           image_size: int = 64, sensor_size: int = 1) -> TorchDatasetSplits:
    """
    Creates the needed objects to use the starcraft dataset
    :param path:                Location to load the starcraft dataset into
    :param targets:             Fields from metadata to use as the regression target, can be a string or list of strings
    :param validation_percent:  Percentage of training data to use for validation
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :param image_format:        Format to use for the dataset, supports 'bag-of-units-first' and 'cifar10'
    :param image_size:          Size of the image in pixels
    :param sensor_size:         Size of sensors for making values missing
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the starcraft dataset"
    # insure the path exists, this is supposed to be done inside the starcraft logic but they don't create parents
    Path(path).mkdir(exist_ok=True, parents=True)

    # ensure targets is always a list
    if targets is None:
        targets = []
    elif isinstance(targets, str):
        targets = [targets]

    # create metadata using
    # TODO: can we reasonably support other image formats? for now hardcoding to 'bag-of-units-first'
    meta = ImageDatasetMeta("starcraft", targets, image_size, sensor_size, 3)

    # we only have train and test for starcraft, so split train into train and validation by percent
    # TODO: consider supporting seeded randomizing the split indices? prevent bias due to ordering in the set
    trainingValidation = createDataset(path, True, image_format, image_size, targets)
    testing = createDataset(path, False, image_format, image_size, targets)

    # handle limits, for quicker testing
    # no extra work if limits are not defined
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

    return TorchDatasetSplits(training, validation, testing, meta)
