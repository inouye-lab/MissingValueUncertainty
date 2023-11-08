from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

import torch
from overrides import override
from torch import Tensor

from .dataset import validateFeatures


def containsMissing(features: Tensor) -> bool:
    """
    Checks if the given features contains any missing values
    :param features:  Input tensor
    :return:  True if any values are missing
    """
    return torch.count_nonzero(torch.isnan(features)) > 0


class Imputator(ABC):
    @abstractmethod
    @property
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

    @override
    @property
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

    @override
    @property
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
