import logging
from pathlib import Path
from typing import Tuple, List, Union, Dict

import torch
from overrides import override
from sc2image import StarCraftImage, StarCraftCIFAR10, StarCraftMNIST
from torch import Tensor
from torch.utils.data import Dataset

from mvu.dataset.specialized.baselines import getTransform
from ..meta import ImageDatasetMeta
from ..torch_utils import TorchDatasetSplits


class StarCraftDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Wrapper around the starcraft image dataset class put the data in the format we expect"""

    base: StarCraftImage
    """Base dataset instance, will fetch data from it"""
    targets: List[str]
    """Metadata variable to use as regression target"""

    def __init__(self, path: str, imageFormat: str, imageSize: int, targets: List[str], train: bool):
        # TODO: support class instead of just targets, probably mutually exclusive
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
        return (unitValues.to(torch.float32) / 127.5) - 1, targets


def _createDataset(path: str, train: bool, transform: callable, image_format: str = 'bag-of-units-first', image_size: int = 64,
                  targets: Union[str, List[str]] = None) -> Dataset:
    if image_format == "cifar10":
        assert targets is not None, "StartCraft CIFAR10 does not support targets"
        return StarCraftCIFAR10(
            root=path,
            train=train,
            download=True,
            transform=transform
        )
    else:
        # TODO: can directly use the original class likely
        return StarCraftDataset(path, image_format, 64, targets, train)


def createStarCraftDataset(path: str = None, targets: Union[str, List[str]] = None,
                           validation_percent: float = 0.3, samples: Dict[str,int] = None,
                           image_format: str = 'bag-of-units-first', normalization: str = "0.5",
                           image_size: int = 64, sensor_size: int = 1, **kwargs) -> TorchDatasetSplits:
    """
    Creates the needed objects to use the starcraft dataset
    :param path:                Location to load the starcraft dataset into
    :param targets:             Fields from metadata to use as the regression target, can be a string or list of strings
    :param validation_percent:  Percentage of training data to use for validation
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :param image_format:        Format to use for the dataset, supports 'bag-of-units-first' and 'cifar10'
    :param normalization:       Normalization method to use for images. Defaults to 0.5 mean and std.
    :param image_size:          Size of the image in pixels
    :param sensor_size:         Size of sensors for making values missing
    :param kwargs:              Additional image transform parameters
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the starcraft dataset"
    # insure the path exists, this is supposed to be done inside the starcraft logic but they don't create parents
    Path(path).mkdir(exist_ok=True, parents=True)

    # ensure targets is always a list
    if targets is None:
        if image_format == "cifar10":
            targets = [f"{map} at {time}" for (map, time) in StarCraftMNIST.classes]
        else:
            targets = []
    elif isinstance(targets, str):
        targets = [targets]
    elif image_format == "cifar10":
        raise NotImplementedError("CIFAR10 does not currently support changing target list")

    logging.info(f"Using {len(targets)} StarCraft targets: {targets}")

    # create metadata using
    # TODO: can we reasonably support other image formats? for now hardcoding to 'bag-of-units-first'
    meta = ImageDatasetMeta("starcraft", targets, image_size, sensor_size, 3)

    # find transform
    transformFunc = getTransform(image_size, 32 if image_format == "cifar10" else 64, normalization=normalization, **kwargs)

    # we only have train and test for starcraft, so split train into train and validation by percent
    # TODO: consider supporting seeded randomizing the split indices? prevent bias due to ordering in the set
    trainingValidation = _createDataset(path, True, transformFunc, image_format, image_size, targets)
    testing = _createDataset(path, False, transformFunc, image_format, image_size, targets)

    return TorchDatasetSplits.split(meta, trainingValidation, testing, validation_percent=validation_percent, samples=samples)
