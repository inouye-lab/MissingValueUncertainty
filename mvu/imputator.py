from abc import ABC, abstractmethod
from typing import Optional

import torch
from overrides import override
from pandas import DataFrame
from statsmodels.imputation.mice import MICEData
from torch import Tensor

from .dataset.meta import validateFeatures, INDEX_SAMPLE, INDEX_FEATURE, DatasetMeta


def containsMissing(features: Tensor) -> bool:
    """
    Checks if the given features contains any missing values
    :param features:  Input tensor
    :return:  True if any values are missing
    """
    return torch.count_nonzero(torch.isnan(features)) > 0


class Imputator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of this imputator for saving in result CSV."""
        pass

    def impute(self, features: Tensor, copy: bool = True) -> Tensor:
        """
        Replaces missing values (that is, NaN values) in the given tensor.
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features=
        :param copy:     If true, copy the tensor before modifying it
        :return: Output tensor, same dimensions as input
        """
        if not containsMissing(features):
            return features
        if copy:
            features = features.clone()
        self._impute(features)
        assert not containsMissing(features), "Imputation did not remove all missing features"
        return features

    @abstractmethod
    def _impute(self, features: Tensor) -> None:
        """
        Replaces missing values (that is, NaN values) in the given tensor.
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features. May be freely modified
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
    def _impute(self, features: Tensor) -> None:
        features[torch.isnan(features)] = 0


class ConstantImputator(Imputator):
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
    def _impute(self, features: Tensor) -> None:
        featureCount = len(self.constant)
        validateFeatures(features, featureCount)
        for i in range(featureCount):
            features[torch.isnan(features[:, i]), i] = self.constant[i]


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

    def _impute(self, features: Tensor) -> None:
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
        mouse = MICEData(DataFrame(inputs.numpy(),
                                   columns=[label.replace(" ", "_") for label in self.metadata.labels]))
        mouse.update_all(self.iterations)

        # pull result into features
        features[:, :] = torch.from_numpy(mouse.next_sample().values[:sampleCount, :])
        self.metadata.normalizeFeatures(features, copy=False)
