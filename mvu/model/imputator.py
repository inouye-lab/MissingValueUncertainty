import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Optional

import torch
from overrides import override
from pandas import DataFrame
from statsmodels.imputation.mice import MICEData
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .common import CachableModel, Namable
from ..dataset.meta import validateFeatures, INDEX_SAMPLE, INDEX_FEATURE, DatasetMeta
from ..serializer import SerializerMixin


def containsMissing(features: Tensor) -> bool:
    """
    Checks if the given features contains any missing values
    :param features:  Input tensor
    :return:  True if any values are missing
    """
    return torch.count_nonzero(torch.isnan(features)) > 0


class Imputator(CachableModel, Namable, ABC):
    def impute(self, features: Tensor, copy: bool = True, rand: Generator = None, indices: Tensor = None) -> Tensor:
        """
        Replaces missing values (that is, NaN values) in the given tensor.
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features=
        :param copy:     If true, copy the tensor before modifying it
        :param rand:     Random state for randomized imputation
        :param indices:  Sample indices for the sake of caching. This should only be used to reduce computation times,
                         not in any way that provides access to normally hidden data.
        :return: Output tensor, same dimensions as input
        """
        if not containsMissing(features):
            return features
        if copy:
            features = features.clone()
        self._impute(features, rand, indices)
        assert not containsMissing(features), "Imputation did not remove all missing features"
        return features

    @abstractmethod
    def _impute(self, features: Tensor, rand: Generator = None, indices: Tensor = None) -> None:
        """
        Replaces missing values (that is, NaN values) in the given tensor.
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features. May be freely modified
        :param indices:  Sample indices for the sake of caching. This should only be used to reduce computation times,
                         not in any way that provides access to normally hidden data.
        :param rand:     Random state for randomized imputation
        """
        pass


class SerializableImputator(Imputator, SerializerMixin, ABC):
    """
    Extension of Imputator that supports serialization. Used for learned imputators such as mean.
    """
    pass


class ZeroImputator(Imputator):
    """
    Imputator that replaces all missing values with zero, provided mainly as a baseline.
    """
    # TODO: consider other value imputators, though that will mess with onehot

    @property
    @override
    def name(self) -> str:
        return "Zero Imputation"

    @override
    def _impute(self, features: Tensor, rand: Generator = None, indices: Tensor = None) -> None:
        features[torch.isnan(features)] = 0


class ConstantImputator(SerializableImputator):
    """
    Imputator that replaces all missing values with a constant.
    """

    constant: Tensor
    """Vector of size (features,) of values to replace missing values"""

    _name: str
    """Name of the constant, e.g. "mean" or "median"."""

    @property
    @override
    def name(self) -> str:
        return f"{self._name} Imputation"

    def __init__(self, constant: Tensor, name: str):
        self.constant = constant
        self._name = name

    @override
    def _impute(self, features: Tensor, rand: Generator = None, indices: Tensor = None) -> None:
        featureCount = len(self.constant)
        validateFeatures(features, featureCount)
        for i in range(featureCount):
            features[torch.isnan(features[:, i]), i] = self.constant[i]

    @classmethod
    def meanFromDataloader(cls, data: DataLoader, showProgress: bool = False, device: torch.device = None) -> "ConstantImputator":
        startTime = perf_counter()

        # will figure out the shape of the mean after seeing the first sample
        means: Optional[Tensor] = None
        numSamples = 0

        # iterate batches, finding the sum of all elements
        batches = len(data)
        for i, (features, targets) in enumerate(data):
            features: Tensor
            if device is not None:
                features = features.to(device)

            # if the mean is not yet started, start it with zeros
            if means is None:
                means = torch.zeros_like(features[0])
                logging.info(f"Output shape: {means.shape}")

            # print progress bar, will overwrite itself with each iteration
            if showProgress:
                print(f"Computing mean batch {i+1:10}/{batches}", end="\r")
            means += features.sum(dim=0)
            numSamples += features.shape[0]

        # convert the sum into the final means
        means /= numSamples
        logging.info(f"Computed dataset mean in {perf_counter() - startTime} seconds")
        return cls(means, name="Mean")


class MiceImputator(Imputator):
    metadata: DatasetMeta
    """Dataset metadata for constructing the dataframe"""
    iterations: int
    """Number of iterations to run"""
    additionalData: Optional[Tensor]
    """Additional data to augment MICE with"""
    augmentName: str
    """Name to use for the data augmentation"""

    def __init__(self, metadata: DatasetMeta, iterations: int, additionalData: Tensor = None, augmentName: str = None):
        self.metadata = metadata
        self.iterations = iterations
        self.additionalData = additionalData
        self.augmentName = augmentName

    @property
    @override
    def name(self) -> str:
        name = f"Mice {self.iterations} Imputation"
        if self.augmentName is not None:
            name += f" - {self.augmentName} Augment"
        return name

    @override
    def _impute(self, features: Tensor, rand: Generator = None, indices: Tensor = None) -> None:
        # short circuit early if not augmented and a whole column of features is missing
        sampleCount = features.shape[INDEX_SAMPLE]
        if self.additionalData is None:
            isMissing = torch.isnan(features)
            invalidFeatures = []
            for i in range(features.shape[INDEX_FEATURE]):
                if torch.count_nonzero(isMissing[:, i]) == sampleCount:
                    invalidFeatures.append(self.metadata.labels[i])
            if len(invalidFeatures) > 0:
                raise ValueError("Non-augmented MICE requires at least 1 sample with each feature, "
                                 f"invalid features {invalidFeatures}")

        # augment data if requested
        inputs = features
        if self.additionalData is not None:
            inputs = torch.concat((features, self.additionalData))

        # run mice
        mouse = MICEData(DataFrame(inputs.cpu().numpy(),
                                   columns=[label.replace(" ", "_") for label in self.metadata.labels]))
        mouse.update_all(self.iterations)

        # pull result into features
        features[:, :] = torch.from_numpy(mouse.next_sample().values[:sampleCount, :])
        self.metadata.normalizeFeatures(features, copy=False)
