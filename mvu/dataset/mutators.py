import torch
from typing import TypeVar, Tuple, Optional

from overrides import override
from torch import Tensor, Generator
from torch.utils.data import Dataset

from mvu.dataset.meta import DatasetMeta, ImageDatasetMeta

T_co = TypeVar('T_co', covariant=True)


class DatasetWrapper(Dataset[T_co]):
    """Standard wrapper for a dataset, to simplify redirecting stuff to base."""

    base: Dataset[T_co]

    def __init__(self, base: Dataset[T_co]):
        self.base = base

    @override
    def __getitem__(self, item):
        return self.base[item]

    # noinspection PyTypeChecker
    def __len__(self):
        return len(self.base)


class MaskedDataset(DatasetWrapper[T_co]):
    """
    Dataset that removes all features matching the passed tensor.
    Expects base to be a tensor dataset of `(features, targets)`.
    """

    featuresToDrop: Tensor
    """Tensor of features to drop. Should be the same dimension as a single sample features"""
    missingValue: float
    """Value to assign to the missing features"""
    includeMask: bool
    """If true, includes the mask in the feature tensor"""
    combineChannels: bool
    """If true, the mask will merge the first dimension into a single value, indicating any channel is missing"""
    returnOriginal: bool
    """If true, returns the original tensor alongside the masked tensor"""

    def __init__(self, base: Dataset[T_co], featuresToDrop: Tensor, missingValue: float = torch.nan,
                 includeMask: bool = False, combineChannels: bool = True, returnOriginal: bool = False):
        super().__init__(base)
        self.featuresToDrop = featuresToDrop
        self.missingValue = missingValue
        self.includeMask = includeMask
        self.combineChannels = combineChannels
        self.returnOriginal = returnOriginal

    @override
    def __getitem__(self, item) -> Tuple[Tensor, ...]:
        data = self.base[item]
        original = data[0]
        features = original.clone()
        features[self.featuresToDrop] = self.missingValue
        if self.includeMask:
            # start with a 1 mask, dropping requested features
            mask = torch.zeros_like(features)
            mask[self.featuresToDrop] = 1
            # next, squeeze it to remove first dimension if requested
            if self.combineChannels:
                mask, _ = mask.max(dim=0, keepdim=True)
                # finally, combine it with the features
            features = torch.cat((features, mask), dim=0)

        # return original tensor if requested, useful for training dirchlets
        # PyRedundantParentheses not supported in python 3.6
        if self.returnOriginal:
            # if we are including the original and including masks, make sure to get the mask in the original
            if self.includeMask:
                mask: Tensor
                if self.combineChannels:
                    mask = torch.zeros_like(original[0]).unsqueeze(0)
                else:
                    mask = torch.zeros_like(original)
                original = torch.cat((original, mask), dim=0)

            # noinspection PyRedundantParentheses
            return (features, original, *data[1:])
        # noinspection PyRedundantParentheses
        return (features, *data[1:])


SpecificFeatureRemovingDataset = MaskedDataset


class FeatureCountRemovingDataset(DatasetWrapper[T_co]):
    """
    Dataset that removes the requested number of features from each sample.
    Expects base to be a tensor dataset of `(features, targets)`.
    For consistency, do not shuffle and only iterate the loader once, later passes may remove different features.
    This was done for simplicity as the usage only requires a single iteration.
    """

    metadata: DatasetMeta
    """Metadata"""
    numToDrop: int
    """Number of features to drop"""
    rand: Generator
    """Generator to remove features."""

    def __init__(self, base: Dataset[T_co], metadata: DatasetMeta, numToDrop: int,
                 rand: Generator = None):
        super().__init__(base)
        self.metadata = metadata
        self.numToDrop = numToDrop
        self.rand = rand

    @override
    def __getitem__(self, item) -> Tuple[Tensor, ...]:
        data = self.base[item]
        if self.numToDrop > 0:
            features = data[0].clone()
            features[self.metadata.sampleDropIndexes(self.numToDrop, rand=self.rand)] = torch.nan
            # noinspection PyRedundantParentheses
            # not supported in python 3.6
            return (features, *data[1:])
        return data


def createMask(meta: Optional[DatasetMeta], name: str, image_size: int = None, channels: int = None) -> Tensor:
    """
    Creates a boolean image mask for use in SpecificFeatureRemovingDataset
    :param meta:         Dataset meta, for populating unset arguments
    :param name:         Mask name, determines which region of the image is missing
    :param image_size:   Size of the image, if none pulls from meta
    :param channels:     Number of channels for the image, if none pulls from meta
    :return:  Boolean tensor mask
    """
    if image_size is None or channels is None:
        assert isinstance(meta, ImageDatasetMeta)
        if image_size is None:
            image_size = meta.imageSize
        if channels is None:
            channels = meta.channels

    mask = torch.zeros((channels, image_size, image_size), dtype=torch.bool)
    if name == "top":
        mask[:, 0:image_size // 2, :] = True
    elif name == "bottom":
        mask[:, image_size // 2:image_size, :] = True

    else:
        raise ValueError(f"Unknown mask name '{name}'")

    return mask
