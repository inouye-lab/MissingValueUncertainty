import gzip
import logging
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, Generator

INDEX_SAMPLE = 0
"""Index of the dimension representing samples"""

INDEX_FEATURE = 1
"""Index of the dimension representing the feature"""


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

    def __init__(self, name, target, labels: List[str], groups: Optional[Tensor]):
        assert groups is None or len(groups) == len(labels), "Labels and groups must be the same size"
        self.name = name
        self.target = target
        self.labels = labels
        self.groups = groups
        self._numGroups = None

    def __str__(self):
        return (f"DatasetMeta{{name: '{self.name}', target: '{self.target}', labels: {str(self.labels)}, "
                f"groups: {str(self.groups)}}}")

    # ditch caches when saving state, see https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        # copy original attributes to avoid breaking object state
        state = self.__dict__.copy()
        # ditch caches
        del state['_numGroups']
        return state

    def __setstate__(self, state):
        # restore instance attributes
        self.__dict__.update(state)
        # ensure caches are set to none, prevents undefined vs none problems
        self._numGroups = None

    def isFeaturesValid(self, features: Tensor) -> bool:
        """
        If true, the given feature set is compatible with this metadata.
        """
        # TODO: consider validating one hot inputs are actually one hot, via groups
        return features.shape[INDEX_FEATURE] == self.numInputs

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
        assert metadata is None or metadata.isFeaturesValid(features), \
            "Inconsistent number of features in metadata and dataset"
        assert features.shape[INDEX_SAMPLE] == targets.shape[INDEX_SAMPLE], "Must have a target for each sample"
        self.features = features
        self.targets = targets
        self.metadata = metadata

    def clone(self) -> "Dataset":
        """Creates a copy of this dataset to allow modifying the tensors (e.g. for missingness)"""
        return Dataset(self.features.clone(), self.targets.clone(), self.metadata)

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


class DatasetSplits(object):
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

    def save(self, path: str) -> None:
        """
        Saves the data to binary
        :param path: Path excluding extension
        """
        with gzip.open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DatasetSplits":
        """
        Loads the splits from binary
        :param path: Path excluding extension
        :return: Loaded dataset
        """
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, cls)
        return data


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
            if len(unique) == 2:
                featureSizes[fIndex] = 1
                df[f"{featureName} {unique[0]}"] = (df[featureName] == unique[0]).astype(float)
            else:
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
