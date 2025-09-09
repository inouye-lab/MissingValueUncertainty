from abc import abstractmethod, ABC
from enum import Enum
from math import ceil
from typing import TypeVar, Tuple, Optional, List, Union

import torch
from overrides import override
from torch import Tensor, Generator
from torch.utils.data import Dataset, random_split, ConcatDataset

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


class IncludeMask(Enum):
    NONE = 0
    """Mask is never included"""
    MISSING = 1
    """Mask is only included for the missing feature"""
    ALWAYS = 2
    """Mask is always included"""

    # noinspection PySimplifyBooleanCheck
    @staticmethod
    def convert(includeMask: Union["IncludeMask",bool]):
        if includeMask == True:
            return IncludeMask.ALWAYS
        if includeMask == False:
            return IncludeMask.NONE
        return includeMask


def fullyObservedMask(original: Tensor, combineChannels: bool = True, dim: int = 0) -> Tensor:
    """
    Adds the mask channel to a fully observed image.
    :param original:         Maskless image.
    :param combineChannels: If true, a single channel is added for the mask. If false, 1 channel per channel in the original image.
    :param dim:             The dimension to add the mask to.
    :return: Image with the mask added
    """
    mask: Tensor
    if combineChannels:
        # map the dimension size to 1, rest to the original size
        mask = torch.zeros(tuple(1 if i == dim else size for i, size in enumerate(original.shape)), device=original.device)
    else:
        mask = torch.zeros_like(original)
    return torch.cat((original, mask), dim=dim)


class MaskedDataset(DatasetWrapper[T_co]):
    """
    Dataset that removes all features matching the passed tensor.
    Expects base to be a tensor dataset of `(features, targets)`.
    """

    featuresToDrop: Tensor
    """Tensor of features to drop. Should be the same dimension as a single sample features"""
    missingValue: float
    """Value to assign to the missing features"""
    includeMask: IncludeMask
    """If true, includes the mask in the feature tensor"""
    combineChannels: bool
    """If true, the mask will merge the first dimension into a single value, indicating any channel is missing"""
    returnOriginal: bool
    """If true, returns the original tensor alongside the masked tensor"""

    def __init__(self, base: Dataset[T_co], featuresToDrop: Tensor, missingValue: float = torch.nan,
                 includeMask: Union[IncludeMask,bool] = IncludeMask.NONE, combineChannels: bool = True, returnOriginal: bool = False):
        super().__init__(base)
        self.featuresToDrop = featuresToDrop
        self.missingValue = missingValue
        self.includeMask = IncludeMask.convert(includeMask)
        self.combineChannels = combineChannels
        self.returnOriginal = returnOriginal

    def _getFeaturesToDrop(self, item):
        return self.featuresToDrop

    @override
    def __getitem__(self, item) -> Tuple[Tensor, ...]:
        data = self.base[item]
        original = data[0]
        features = original.clone()
        featuresToDrop = self._getFeaturesToDrop(item)
        features[featuresToDrop] = self.missingValue
        if self.includeMask != IncludeMask.NONE:
            # start with a 1 mask, dropping requested features
            mask = torch.zeros_like(features)
            mask[featuresToDrop] = 1
            # next, squeeze it to remove first dimension if requested
            if self.combineChannels:
                mask, _ = mask.max(dim=0, keepdim=True)
                # finally, combine it with the features
            features = torch.cat((features, mask), dim=0)

        # return original tensor if requested, useful for training dirchlets
        # PyRedundantParentheses not supported in python 3.6
        if self.returnOriginal:
            # if we are including the original and including masks, make sure to get the mask in the original
            if self.includeMask == IncludeMask.ALWAYS:
                original = fullyObservedMask(original, self.combineChannels)

            # noinspection PyRedundantParentheses
            return (features, original, *data[1:])
        # noinspection PyRedundantParentheses
        return (features, *data[1:])


SpecificFeatureRemovingDataset = MaskedDataset


class RandomMaskedDataset(MaskedDataset):
    """
    Extension of `RandomMaskedDataset` which randomly removes one of a list of masks.
    """

    masks: List[Tensor]
    """List of random masks to choose between"""
    rand: Generator
    """Generator for mask selection"""

    def __init__(self, base: Dataset[T_co], masks: List[Tensor], rand: Generator = None, *args, **kwargs):
        super().__init__(base, masks[0], *args, **kwargs)
        assert len(masks) > 1, "Must have at least two masks to do a random mask"
        self.masks = masks
        self.rand = rand

    @override
    def _getFeaturesToDrop(self, item):
        return self.masks[torch.randint(len(self.masks), (1,), generator=self.rand)]


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


class BlockRemovingDataset(MaskedDataset, ABC):
    """Dataset that removes random blocks from the image"""

    imageSize: int
    """Image size in pixels"""
    sensorSize: int
    """Size of each block"""
    channels: int
    """Number of image channels"""

    totalBlocks: int
    """Number of blocks in the image"""
    rand: Optional[Generator]
    """Random state"""
    groups: Tensor
    """Integers assigning each pixel in the 3 channels to a block index"""

    def __init__(self, base: Dataset[T_co], imageSize: int, sensorSize: int, channels: int,
                 *args,
                 rand: Optional[Generator] = None,
                 **kwargs):
        super().__init__(base, torch.empty((1,)), *args, **kwargs)
        self.imageSize = imageSize
        self.sensorSize = sensorSize
        self.channels = channels
        self.rand = rand

        # prepare details for dropping
        self.sensorsPerAxis = ceil(imageSize / sensorSize)
        self.totalBlocks = self.sensorsPerAxis ** 2

    @classmethod
    def fromMetadata(cls, base: Dataset[T_co], meta: DatasetMeta, *args, **kwargs):
        """Constructs this dataset using an image metadata"""
        if not isinstance(meta, ImageDatasetMeta):
            raise TypeError(f"Expected ImageDatasetMeta, got {type(meta)}")
        return cls(base, meta.imageSize, meta.sensorSize, meta.channels, *args, **kwargs)

    def blocksToImage(self, blocks: Tensor) -> Tensor:
        """
        Converts the given tensor of the number of blocks into an image of the appropriate size
        """
        return (blocks.reshape(self.sensorsPerAxis, self.sensorsPerAxis)
         # repeat across both axes, then crop to image size (in case of non-square)
         .repeat_interleave(self.sensorSize, dim=0)
         .repeat_interleave(self.sensorSize, dim=1)[0:self.imageSize, 0:self.imageSize]
         # finally, add in the number of channels
         .reshape(-1, self.imageSize, self.imageSize).repeat_interleave(self.channels, dim=0))

    @abstractmethod
    @override
    def _getFeaturesToDrop(self, item) -> Tensor:
        pass


class FixedBlockRemovingDataset(BlockRemovingDataset):
    _dropIndices: Tensor
    """Indices to drop"""
    _dropFeatures: Tensor
    """Features to drop"""

    def __init__(self, base: Dataset[T_co], imageSize: int, sensorSize: int, channels: int, dropIndices: Tensor, *args, **kwargs):
        super().__init__(base, imageSize, sensorSize, channels, *args, **kwargs)
        self._dropIndices = dropIndices
        self._dropFeatures = self.blocksToImage(dropIndices)

    @property
    def dropIndices(self) -> Tensor:
        return self._dropIndices

    @dropIndices.setter
    def dropIndices(self, drop: Tensor) -> None:
        self._dropIndices = drop
        self._dropFeatures = self.blocksToImage(drop)

    @override
    def _getFeaturesToDrop(self, item) -> Tensor:
        return self._dropFeatures


class BlockCountRemovingDataset(BlockRemovingDataset):
    """Block removing dataset that removes blocks with a uniform sampled count"""

    featureWeights: Tensor
    """Tensor of 1s of the proper size for multinomial sampling"""

    dropMin: int
    """Minimum number of blocks to drop"""
    dropMax: int
    """Maximum number of blocks to drop"""

    all: Tensor
    """Mask representing all features dropped"""
    none: Tensor
    """Mask representing no features dropped"""

    def __init__(self, base: Dataset[T_co], *args,
                 dropMin: int = 0, dropMax: int = None,
                 **kwargs):
        super().__init__(base, *args, **kwargs)

        # use total blocks for max if unset
        if dropMax is None:
            dropMax = self.totalBlocks
        assert 0 <= dropMin <= dropMax <= self.totalBlocks, "Min must be less than max, and both between 0 and the number of blocks"
        self.dropMin = dropMin
        self.dropMax = dropMax
        self.none = torch.zeros((self.channels, self.imageSize, self.imageSize), dtype=torch.bool)
        self.all = torch.ones((self.channels, self.imageSize, self.imageSize), dtype=torch.bool)

        # weights for multinomial
        self.featureWeights = torch.ones(self.totalBlocks)

        # construct grid of sensor index for each location
        # start with a list from 0 to N, reshape into a grid
        self.groups = self.blocksToImage(torch.arange(0, self.totalBlocks))

    def _getBlocksToDrop(self) -> int:
        """Gets the number of blocks to drop"""
        if self.dropMin == self.dropMax:
            return self.dropMin
        # TODO: seeding?
        return torch.randint(self.dropMin, self.dropMax + 1, (1,)).item()

    @override
    def _getFeaturesToDrop(self, item) -> Tensor:
        # find out how many to drop, can skip multinomial if none or all
        numToDrop = self._getBlocksToDrop()
        if numToDrop <= 0:
            return self.none
        if numToDrop >= self.totalBlocks:
            return self.all

        # select enough indices to drop
        # TODO: seeding?
        dropIndexes = torch.multinomial(self.featureWeights, numToDrop, replacement=False, generator=self.rand)
        # construct mask by any that match
        return torch.isin(self.groups, dropIndexes)


class BlockDropoutDataset(BlockRemovingDataset):
    """Block removing dataset which has a chance to drop each block"""

    dropChances: Tensor

    def __init__(self, base: Dataset[T_co], imageSize: int, sensorSize: int, channels: int, *args,
                 dropChance: float = 0.5, cleanChance: float = 0,
                 **kwargs):
        super().__init__(base, imageSize, sensorSize, channels, *args, **kwargs)
        assert 0 < dropChance < 1, "Drop chance must be a percentage between 0 and 1"

        self.dropChances = torch.tensor([dropChance] * self.totalBlocks, dtype=torch.float)
        self.cleanChance = cleanChance

        if cleanChance > 0:
            self.none = torch.zeros((self.channels, self.imageSize, self.imageSize), dtype=torch.bool)

    @override
    def _getFeaturesToDrop(self, item) -> Tensor:
        if self.cleanChance > 0 and torch.rand((), generator=self.rand) < self.cleanChance:
            return self.none
        return self.blocksToImage(torch.bernoulli(self.dropChances, generator=self.rand).to(torch.bool))

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
    elif name == "full":
        mask[:] = True

    elif name != "none":
        raise ValueError(f"Unknown mask name '{name}'")

    return mask


def splitDataset(dataset: Dataset, splits: int, rand: Generator = None) -> List[Dataset]:
    """
    Splits the dataset into the given number of parts
    :param dataset:  Dataset to split
    :param splits:   Number of subsets to create
    :param rand:     Random state
    :return:  List of split datasets
    """
    # noinspection PyTypeChecker
    totalLength = len(dataset)
    subsetLength = totalLength // splits
    lengths = [subsetLength] * splits
    # Adjust the last part to account for any remainder if the dataset is not divisible by n
    remainder = totalLength % splits
    for i in range(remainder):
        lengths[i] += 1

    return random_split(dataset, lengths, generator=rand)


def distributeMasks(dataset: Dataset, masks: List[Tensor], rand: Generator = None, *args, **kwargs) -> Dataset:
    """
    Divides the dataset randomly between the list of masks
    :param dataset:  Dataset to divide
    :param masks:    List of masks
    :param rand:     Random state
    :return:  Dataset with masks applied to each part
    """
    subsets = splitDataset(dataset, len(masks), rand)
    masked = [MaskedDataset(subset, mask, *args, **kwargs) for subset, mask in zip(subsets, masks)]
    return ConcatDataset(masked)


def randomDropping(dataset: Dataset, meta: DatasetMeta, name: str = "", **kwargs) -> Dataset:
    """Creates a random dropping dataset with the given parameters"""

    if name == "block-count":
        return BlockCountRemovingDataset.fromMetadata(dataset, meta, **kwargs)
    elif name == "block-dropout":
        return BlockDropoutDataset.fromMetadata(dataset, meta, **kwargs)
    else:
        raise ValueError(f"Unknown random dropping method name '{name}'")