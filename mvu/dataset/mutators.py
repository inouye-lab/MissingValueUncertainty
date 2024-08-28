import torch
from typing import TypeVar, Tuple

from overrides import override
from torch import Tensor, Generator
from torch.utils.data import Dataset

from mvu.dataset.torch import TwoTensor
from mvu.dataset.meta import DatasetMeta

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


class SpecificFeatureRemovingDataset(DatasetWrapper[TwoTensor]):
    """
    Dataset that removes all features matching the passed tensor.
    Expects base to be a tensor dataset of `(features, targets)`.
    """

    featuresToDrop: Tensor
    """Tensor of features to drop"""

    def __init__(self, base: Dataset[TwoTensor], featuresToDrop: Tensor):
        super().__init__(base)
        self.featuresToDrop = featuresToDrop

    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        features, targets = self.base[item]
        features = features.clone()
        features[self.featuresToDrop] = torch.nan
        return features, targets


class FeatureCountRemovingDataset(DatasetWrapper[TwoTensor]):
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

    def __init__(self, base: Dataset[TwoTensor], metadata: DatasetMeta, numToDrop: int,
                 rand: Generator = None):
        super().__init__(base)
        self.metadata = metadata
        self.numToDrop = numToDrop
        self.rand = rand

    def __getitem__(self, item):
        features, targets = self.base[item]
        if self.numToDrop > 0:
            features = features.clone()
            features[self.metadata.sampleDropIndexes(self.numToDrop, rand=self.rand)] = torch.nan
        return features, targets
