import math
import os
from typing import List, Optional

import torch
from overrides import override
from torch import Tensor, Generator

INDEX_SAMPLE = 0
"""Index of the dimension representing samples"""
INDEX_FEATURE = 1
"""Index of the dimension representing the feature"""


def validateFeatures(features: Tensor, expectedFeatures: int, isVector: bool = False):
    """
    Validates the matrix is a valid features matrix for the given size
    :param features:          Matrix to validate dimensions
    :param expectedFeatures:  Expected number of features
    :param isVector:          If true, features is a feature vector of size `(features,)`.
                              If false it's a matrix size `(samples, features)`.
    """
    featureSize: int
    if isVector:
        featureSize = len(features)
    else:
        featureSize = features.shape[INDEX_FEATURE]
    assert featureSize == expectedFeatures, \
        f"Expected feature dimension to be {expectedFeatures}, found {featureSize}"


class DatasetMeta(object):
    """Object representing non-numeric data in a dataset, for explanation purposes mainly"""

    name: str
    """Friendly readable name of the dataset"""

    target: str
    """Name of the target feature"""

    labels: List[str]
    """List of labels for each feature index"""

    groups: Optional[Tensor]
    """
    Group indexes for categorical features. Ranges from 0 to N-1 where N is the number of distinct features.
    If None, every feature is considered distinct.
    """

    _numGroups: Optional[int]
    """Cached number of features, computed from groups"""

    _featureWeights: Optional[torch.Tensor]
    """Feature weights for random feature drops"""

    def __init__(self, name: str, target: str, labels: List[str], groups: Optional[Tensor]):
        assert groups is None or len(groups) == len(labels), "Labels and groups must be the same size"
        self.name = name
        self.target = target
        self.labels = labels
        self.groups = groups
        self._numGroups = None
        self._featureWeights = None

    def __str__(self):
        return (f"DatasetMeta{{name: '{self.name}', target: '{self.target}', labels: {str(self.labels)}, "
                f"groups: {str(self.groups)}}}")

    # ditch caches when saving state, see https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        # copy original attributes to avoid breaking object state
        state = self.__dict__.copy()
        # ditch caches
        del state['_numGroups']
        del state['_featureWeights']
        return state

    def __setstate__(self, state):
        # restore instance attributes
        self.__dict__.update(state)
        # ensure caches are set to none, prevents undefined vs none problems
        self._numGroups = None
        self._featureWeights = None

    def validateFeatures(self, features: Tensor, isVector: bool = False) -> None:
        """Runs assertions to ensure the feature matrix is valid"""
        validateFeatures(features, self.numInputs, isVector)

    @property
    def numInputs(self) -> int:
        """Gets the input dimension of compatible features, determines size of second dimension of features"""
        return len(self.labels)

    @property
    def numGroups(self) -> int:
        """Gets the number of distinct features in the dataset (as some features are onehot), used for missingness"""
        if self._numGroups is None:
            if self.groups is None:
                self._numGroups = self.numInputs
            else:
                self._numGroups = torch.max(self.groups).item() + 1
        return self._numGroups

    def sampleDropIndexes(self, numToDrop: int, rand: Generator) -> Tensor:
        """
        Samples a boolean Tensor same size as the features of features to drop
        :param numToDrop:   Number of features to drop
        :param rand:        Rand state
        :return:            Index tensor size of `numInputFeatures`
                            Will be either ints or booleans based on if self.groups is None
        """
        if self._featureWeights is None:
            self._featureWeights = torch.ones(self.numGroups)
        # select feature indexes to drop
        dropIndexes = torch.multinomial(self._featureWeights, numToDrop, replacement=False, generator=rand)
        # if we have groups, need to expand that as some features have multiple indexes
        # if we don't have groups, indexes are sufficient
        if self.groups is None:
            return dropIndexes
        return torch.isin(self.groups, dropIndexes)

    def dropCount(self, features: Tensor, numToDrop: int, bySample: bool = True, copy: bool = True,
                  rand: Generator = None) -> Tensor:
        """
        Drops the given number of features from the input tensor.
        :param features:   Input tensor of size `(samples, features)`.
        :param numToDrop:  Number of features to drop, cannot be greater than `numDistinctFeatures`
        :param bySample:   If true, samples features to remove per sample. False drops same in all samples.
        :param copy:       If true, copies the tensor before modifying it
        :param rand:       Rand state
        :return: Tensor with the given features dropped
        """
        self.validateFeatures(features)
        assert 0 <= numToDrop <= self.numGroups, "Cannot drop more features than present in the tensor"
        if numToDrop == 0:
            return features
        if copy:
            features = features.clone()
        if bySample:
            for i in range(features.shape[INDEX_SAMPLE]):
                features[i, self.sampleDropIndexes(numToDrop, rand)] = torch.nan
        else:
            features[:, self.sampleDropIndexes(numToDrop, rand)] = torch.nan
        return features

    def dropSpecified(self, features: Tensor, featuresToDrop: Tensor, copy: bool = True):
        """
        Drops the specified features from the input tensor.
        :param features:   Input tensor of size `(samples, features)`.
        :param featuresToDrop: Index tensor specifying features to drop,
                               typically will be boolean tensor of size `(features,)`
        :param copy: If true, copies the dataset before modifying it. Will not copy if numToDrop is 0
        :return:
        """
        self.validateFeatures(features)
        if copy:
            features = features.clone()
        features[:, featuresToDrop] = torch.nan
        return features

    def normalizeFeatures(self, features: Tensor, copy: bool = True) -> Tensor:
        """
        Ensures all one hot features have a single "hot" value.
        Important as neural networks trained on boolean values will not automatically support probability inputs.
        :param features: Features tensor
        :param copy:     If true, copies the tensor before modifying
        :return: Normalized tensor
        """
        if self.groups is None:
            return features

        vectorInput = len(features.shape) == 1
        if vectorInput:
            features = features.reshape(1, -1)
        self.validateFeatures(features)
        if copy:
            features = features.clone()
        for group in range(self.numGroups):
            # one hot features are any features with at least 2 members in the group
            # noinspection PyTypeChecker
            groupIndexes: Tensor = self.groups == group
            groupSize = torch.count_nonzero(groupIndexes).item()
            if groupSize > 1:
                # argmax finds the most significant feature in each sample
                maxIndexes = torch.argmax(features[:, groupIndexes], dim=INDEX_FEATURE)
                # overwrite probability with onehot
                features[:, groupIndexes] = torch.nn.functional.one_hot(maxIndexes, groupSize).type(torch.float)
        if vectorInput:
            return features.reshape(-1)
        return features

    def countDistinctFeatures(self, indexes: Tensor) -> int:
        """
        Counts the number of distinct features referred to by the given index vector.
        :param indexes:   Index vector, should be a valid index to a feature tensor.
        :return:  Number of distinct features.
        """
        if self.groups is None:
            if indexes.dtype == torch.bool:
                return torch.count_nonzero(indexes).item()
            else:
                return len(torch.unique(indexes))
        return len(torch.unique(self.groups[indexes]))

    def featureName(self, index: int) -> str:
        """
        Gets the name of the feature at the given index
        :param index:  Index of the feature
        :return: Name of the feature
        """
        assert 0 <= index < self.numGroups

        # no groups, just fetch the label
        if self.groups is None:
            return self.labels[index]
        # if groups defined, fetch matching indices
        indices = torch.eq(self.groups, index)
        # assumption: all features in a group are adjacent
        count = torch.count_nonzero(indices).item()  # number of elements in this feature
        first = indices.int().argmax()                     # first index of the feature
        if count == 1:
            return self.labels[first]
        else:
            return os.path.commonprefix(self.labels[slice(first, first+count)]).rstrip()
