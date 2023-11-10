import logging
from typing import List, Optional

import pandas as pd
import torch
from torch import Tensor, Generator

from .serializer import SerializerMixin

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

    def __init__(self, name, target, labels: List[str], groups: Optional[Tensor]):
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

    def _sampleDropIndexes(self, numToDrop: int, rand: Generator) -> Tensor:
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
                features[i, self._sampleDropIndexes(numToDrop, rand)] = torch.nan
        else:
            features[:, self._sampleDropIndexes(numToDrop, rand)] = torch.nan
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


class Dataset(object):
    """Object representing a single dataset of features and targets. Provides guarantee the feature count matches"""

    features: Tensor
    """"X values" in the dataset, dimensions (sample count, feature index)"""

    targets: Tensor
    """"Y values" in the dataset, dimensions (sample count,)"""

    metadata: Optional[DatasetMeta]
    """Metadata in the dataset"""

    def __init__(self, features: Tensor, targets: Tensor, metadata: DatasetMeta = None):
        # Same number of samples ensures every sample has a target
        if metadata is not None:
            metadata.validateFeatures(features)
        assert features.shape[INDEX_SAMPLE] == targets.shape[INDEX_SAMPLE], "Must have a target for each sample"
        self.features = features
        self.targets = targets
        self.metadata = metadata

    def clone(self, cloneTargets: bool = True) -> "Dataset":
        """Creates a copy of this dataset to allow modifying the tensors (e.g. for missingness)"""
        targets = self.targets
        if cloneTargets:
            targets = targets.clone()
        return Dataset(self.features.clone(), targets, self.metadata)

    @property
    def numSamples(self):
        """Gets the number of samples in this datasset, determines size of labels and first dimension of features"""
        return self.features.shape[INDEX_SAMPLE]

    @property
    def numInputs(self):
        """Gets the input dimension of the features, determines size of second dimension of features"""
        # TODO: generalize to allow features to be multidimensional?
        return self.features.shape[INDEX_FEATURE]

    @property
    def numGroups(self):
        """Gets the number of distinct features in the dataset (as some features are onehot), used for missingness"""
        if self.metadata is None:
            return self.numInputs
        return self.metadata.numGroups

    def split(self, indexes: Tensor) -> "Dataset":
        """Creates a new dataset from the given set of indexes"""
        return Dataset(self.features[indexes, :], self.targets[indexes], self.metadata)

    def isSameSet(self, other: "Dataset"):
        """Checks if the given datasets represent the same dataset"""
        return self.metadata is other.metadata and self.numInputs is other.numInputs

    def dropCount(self, numToDrop: int, bySample: bool = True, copy: bool = True,
                  rand: Generator = None) -> "Dataset":
        """
        Drops the given number of features from the input tensor.
        :param numToDrop:  Number of features to drop, cannot be greater than `numDistinctFeatures`
        :param copy:       If true, copies the dataset before modifying it. Will not copy if numToDrop is 0
        :param bySample:   If true, samples features to remove per sample. False drops same in all samples.
        :param rand:       Rand state
        :return: Tensor with the given features dropped
        """
        assert self.metadata is not None, "Cannot drop features without metadata"
        if numToDrop == 0:
            return self
        dataset = self
        if copy:
            dataset = dataset.clone(cloneTargets=False)  # not changing targets
        dataset.metadata.dropCount(dataset.features, numToDrop, bySample, False, rand)
        return dataset

    def dropSpecified(self, featuresToDrop: Tensor, copy: bool = True):
        """
        Drops the specified features from the input tensor.
        :param featuresToDrop: Index tensor specifying features to drop,
                               typically will be boolean tensor of size `(features,)`
        :param copy: If true, copies the dataset before modifying it. Will not copy if numToDrop is 0
        :return:
        """
        assert self.metadata is not None, "Cannot drop features without metadata"
        if torch.count_nonzero(featuresToDrop) == 0:
            return self
        dataset = self
        if copy:
            dataset = dataset.clone(cloneTargets=False)  # not changing targets
        dataset.metadata.dropSpecified(dataset.features, featuresToDrop, False)
        return dataset


class DatasetSplits(SerializerMixin):
    """Object representing a train, validation, and testing split"""

    train: Dataset
    """Dataset used for learning the model"""

    validate: Dataset
    """Dataset used for learning hyperparameters"""

    test: Dataset
    """Dataset used for validating results"""

    metadata: Optional[DatasetMeta]
    """Metadata in the dataset"""

    def __init__(self, train: Dataset, validate: Dataset, test: Dataset):
        # Same number of features ensures they are all representing the same dataset
        assert train.isSameSet(validate)
        assert train.isSameSet(test)

        self.train = train
        self.validate = validate
        self.test = test
        self.metadata = train.metadata

    def clone(self) -> "DatasetSplits":
        """Creates a copy of this dataset to allow modifying the tensors (e.g. for missingness)"""
        return DatasetSplits(self.train.clone(), self.validate.clone(), self.test.clone())


def import_from_csv(name: str, csv: str, targetFeature: str,
                    numericFeatures: List[str], categoricalFeatures: List[str]) -> Dataset:
    """
    Loads the given dataset by name.
    :param name: Name of the dataset, determines the name of the cached binary of data
    :param csv: Path to the dataset CSV
    :param targetFeature: Feature to use as the target
    :param numericFeatures:      List of numeric value features to include
    :param categoricalFeatures:  List of non-numeric value features to include, includes booleans
    :return:  Dataset loaded from the given CSV
    """

    df = pd.read_csv(csv)
    logging.info(f'Loaded {name} dataframe with shape: {df.shape} and columns {df.columns}')

    # start by fetching the labels
    targets = torch.Tensor(df[targetFeature].values)

    # start by dropping features we don't care about
    df = df[numericFeatures + categoricalFeatures]

    # next, convert discrete features to one
    featureGroups = None
    if len(categoricalFeatures) > 0:
        # keep track of elements in each feature
        featureSizes = [0]*len(categoricalFeatures)
        for fIndex, featureName in enumerate(categoricalFeatures):
            unique = df[featureName].unique()
            #if len(unique) == 2:
            #    featureSizes[fIndex] = 1
            #    df[f"{featureName} {unique[0]}"] = (df[featureName] == unique[0]).astype(float)
            #else:
            # TODO: consider if its worth merging true and false for a single boolean into one feature
            # in theory, its easier for the NN to handle separately
            # however the main reason I did not was it makes it harder to identify boolean features for normalizing
            featureSizes[fIndex] = len(unique)
            for value in unique:
                df[f"{featureName} {value}"] = (df[featureName] == value).astype(float)

        df = df.drop(columns=categoricalFeatures)

        # store indexes to keep track of groups of features
        featureGroups = torch.arange(0, len(df.columns))
        # inputIndex is the offset for the first discrete feature
        # groupIndex is the current index in the features group tensor
        inputIndex = groupIndex = len(numericFeatures)
        for fIndex, size in enumerate(featureSizes):
            for i in range(size):
                featureGroups[groupIndex+i] = inputIndex+fIndex
            groupIndex += size

    logging.info(f'After preprocessing, shape: {df.shape} and columns {df.columns}')

    # finally, make final feature matrix
    features = torch.Tensor(df.values)

    # construct metadata
    datasetMeta = DatasetMeta(name, targetFeature, df.columns.tolist(), featureGroups)
    logging.info(f'Final metadata, {datasetMeta}')
    return Dataset(features, targets, datasetMeta)


def split_dataset(dataset: Dataset, validPercent: float = 0.2, testPercent: float = 0.3, rand: Generator = None
                  ) -> DatasetSplits:
    """
    Splits the given dataset based on the given percentages
    :param dataset:       Un-split dataset
    :param validPercent:  Percentage of the total data to use for validation, must be between 0 and 1
    :param testPercent:   Percentage of the total data to use for testing, must be between 0 and 1
    :param rand:          Random generator to permute the input data, if unset no permutation is performed
    :return:  Dataset with each of the given splits
    """
    assert 0 < validPercent < 1, "validPercent must be a percentage"
    assert 0 < testPercent < 1, "testPercent must be a percentage"
    trainPercent = 1 - validPercent - testPercent
    assert 0 < trainPercent, "Total percent cannot be more than 100%"

    logging.info(f"Initial dataset samples: {dataset.numSamples}")

    # permute rows
    samples = dataset.numSamples
    indexes: Tensor
    if rand is not None:
        indexes = torch.randperm(samples, generator=rand)
    else:
        indexes = torch.arange(samples)

    # decide end points for splits
    trainEnd = int(trainPercent * samples)
    validEnd = int(validPercent * samples) + trainEnd

    # create the actual splits
    train = dataset.split(indexes[:trainEnd])
    valid = dataset.split(indexes[trainEnd:validEnd])
    test = dataset.split(indexes[validEnd:])

    logging.info(f"Final samples: Train {train.numSamples}, Validate {valid.numSamples}, Test {test.numSamples}")

    # finally, return
    return DatasetSplits(train, valid, test)
