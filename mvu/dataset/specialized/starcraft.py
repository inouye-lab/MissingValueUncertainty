import logging
from typing import Tuple

from overrides import override
from sc2image import StarCraftImage
from torch import Tensor
from torch.utils.data import Dataset

from mvu.dataset.meta import ImageDatasetMeta
from mvu.dataset.torch import TorchDatasetSplits


class StarCraftDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Wrapper around the starcraft image dataset class put the data in the format we expect"""

    base: StarCraftImage
    """Base dataset instance, will fetch data from it"""
    target: str
    """Metadata variable to use as regression target"""

    def __init__(self, path: str, imageFormat: str, imageSize: int, target: str, train: bool):
        self.base = StarCraftImage(path, image_format=imageFormat, image_size=imageSize, train=train,
                                   return_dict=True, use_metadata_cache=True, download=True)
        self.target = target
        if target is None:
            logging.warning("No target for StarCraft dataset, this will give unexpected behavior in regression tasks.")

    def __len__(self):
        return len(self.base)

    @override
    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        (unitIds, unitValues), data = self.base[item]
        # if no target is defined, just use 0. This just makes it simpler in contexts that are not regressing
        if self.target is None:
            target = 0
        else:
            target = data["metadata"][self.target]
        # expect vectors for the input instead of an image
        # we also convert to floats so we can actually place nan in the tensor for missingness
        # TODO: should we be converting to a range of -1 to 1 instead of 0 to 1?
        return (unitValues.float() / 255).reshape(-1), Tensor([target])


def createStarCraftDataset(path: str = None, target: str = None,
                           image_size: int = 64, sensor_size: int = 1) -> TorchDatasetSplits:
    """
    Creates the needed objects to use the starcraft dataset
    :param path:         Location to load the starcraft dataset into
    :param target:       Field from metadata to use as the regression target
    :param image_size:   Size of the image in pixels
    :param sensor_size:  Size of sensors for making values missing
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the starcraft dataset"

    # create metadata using
    # TODO: can we reasonably support other image formats? for now hardcoding to 'bag-of-units-first'
    imageFormat = 'bag-of-units-first'
    meta = ImageDatasetMeta("starcraft", target, image_size, sensor_size, 3)
    training = StarCraftDataset(path, imageFormat, image_size, target, train=True)
    testing = StarCraftDataset(path, imageFormat, image_size, target, train=False)
    # TODO: validation data
    return TorchDatasetSplits(training, training, testing, meta)