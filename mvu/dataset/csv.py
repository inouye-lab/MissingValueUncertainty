import logging
from typing import Optional, List

import pandas as pd
import torch
from torch import Tensor, Generator
from torch.utils.data import TensorDataset

from .torch import TorchDatasetSplits
from ..dataset.meta import DatasetMeta, INDEX_SAMPLE, INDEX_FEATURE
from ..serializer import SerializerMixin


class CsvDataset(object):
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

    def clone(self, cloneTargets: bool = True) -> "CsvDataset":
        """Creates a copy of this dataset to allow modifying the tensors (e.g. for missingness)"""
        targets = self.targets
        if cloneTargets:
            targets = targets.clone()
        return CsvDataset(self.features.clone(), targets, self.metadata)

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

    def split(self, indexes: Tensor) -> "CsvDataset":
        """Creates a new dataset from the given set of indexes"""
        return CsvDataset(self.features[indexes, :], self.targets[indexes], self.metadata)

    def isSameSet(self, other: "CsvDataset"):
        """Checks if the given datasets represent the same dataset"""
        return self.metadata is other.metadata and self.numInputs is other.numInputs

    def dropCount(self, numToDrop: int, bySample: bool = True, copy: bool = True,
                  rand: Generator = None) -> "CsvDataset":
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

    def toTorch(self) -> TensorDataset:
        """Converts this dataset into a torch dataset."""
        return TensorDataset(self.features, self.targets)


class CsvDatasetSplits(SerializerMixin):
    """Object representing a train, validation, and testing split"""

    train: CsvDataset
    """Dataset used for learning the model"""

    validate: CsvDataset
    """Dataset used for learning hyperparameters"""

    test: CsvDataset
    """Dataset used for validating results"""

    metadata: Optional[DatasetMeta]
    """Metadata in the dataset"""

    def __init__(self, train: CsvDataset, validate: CsvDataset, test: CsvDataset):
        # Same number of features ensures they are all representing the same dataset
        assert train.isSameSet(validate)
        assert train.isSameSet(test)

        self.train = train
        self.validate = validate
        self.test = test
        self.metadata = train.metadata

    def clone(self) -> "CsvDatasetSplits":
        """Creates a copy of this dataset to allow modifying the tensors (e.g. for missingness)"""
        return CsvDatasetSplits(self.train.clone(), self.validate.clone(), self.test.clone())

    def toTorch(self) -> TorchDatasetSplits:
        """Creates an object containing torch datasets, which is what we use primarily outside dataset creation"""
        return TorchDatasetSplits(self.train.toTorch(), self.validate.toTorch(), self.test.toTorch(), self.metadata)


def import_from_csv(name: str, csv: str, targetFeature: str,
                    numericFeatures: List[str], categoricalFeatures: List[str]) -> CsvDataset:
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
    targets = torch.tensor(df[targetFeature].values, dtype=torch.float)

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
    features = torch.tensor(df.values, dtype=torch.float)

    # construct metadata
    datasetMeta = DatasetMeta(name, targetFeature, df.columns.tolist(), featureGroups)
    logging.info(f'Final metadata, {datasetMeta}')
    return CsvDataset(features, targets, datasetMeta)


def split_dataset(dataset: CsvDataset, validPercent: float = 0.2, testPercent: float = 0.3, rand: Generator = None
                  ) -> CsvDatasetSplits:
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
    return CsvDatasetSplits(train, valid, test)
