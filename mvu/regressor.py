import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from overrides import override
from sklearn.linear_model import Ridge
from torch import Tensor
from torch.nn import Module

from .dataset import DatasetSplits, Dataset
from .serializer import SerializerMixin


class Regressor(SerializerMixin, ABC):
    """Base class for the primary regressor for the regression task."""

    @abstractmethod
    def predict(self, features: Tensor) -> Tensor:
        """
        Makes a prediction using this regressor
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features
        :return: Output tensor, with a single dimension representing the prediction
        """
        pass

    @classmethod
    def _processPostLoad(cls, data: Any) -> "Regressor":
        if isinstance(data, Ridge):
            return RidgeRegressor(data)
        if isinstance(data, Module):
            return NeuralNetworkRegressor(data)
        return data

    def evaluateDataset(self, dataset: Dataset, name: str) -> None:
        """
        Evaluates the model on the given dataset and logs the final MSE
        :param dataset:  Dataset to evaluate
        :param name:     Name to print in the log
        """
        predicted = self.predict(dataset.features)
        mse = torch.mean((predicted - dataset.targets) ** 2)
        logging.info(f"MSE for {name}: {mse}")

    def evaluateSplits(self, ds: DatasetSplits) -> None:
        """
        Evaluates the model on the given dataset splits and logs the final MSE
        :param ds: Dataset splits to evaluate
        """
        self.evaluateDataset(ds.train, "train")
        self.evaluateDataset(ds.validate, "validate")
        self.evaluateDataset(ds.test, "test")


class RidgeRegressor(Regressor):
    """Regressor implemented using the SKLearn Ridge Regression functionality"""

    ridge: Ridge

    def __init__(self, ridge: Ridge):
        self.ridge = ridge

    @override
    def predict(self, features: Tensor) -> Tensor:
        return Tensor(self.ridge.predict(features.numpy()))


class NeuralNetworkRegressor(Regressor):
    """Regressor using a torch neural network"""

    nn: Module

    def __init__(self, nn: Module):
        self.nn = nn

    @override
    def predict(self, features: Tensor) -> Tensor:
        return self.nn(features)
