import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from overrides import override
from sklearn.linear_model import Ridge
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from ..dataset.csv import CsvDataset, CsvDatasetSplits
from ..serializer import SerializerMixin


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

    def evaluateDataset(self, dataset: CsvDataset) -> Tensor:
        """
        Evaluates the model on the given dataset and logs the final MSE
        :param dataset:  Dataset to evaluate
        :return:  Mean squared error for the dataset
        """
        predicted = self.predict(dataset.features)
        return torch.mean((predicted - dataset.targets) ** 2)

    @torch.no_grad()
    def evaluateDataloader(self, data: DataLoader, device: torch.device = None) -> Tensor:
        """
        Evaluates the model on the given dataset and logs the final MSE
        :param data:   Data to evaluate
        :param device: Device to use for computation
        :return:  Mean squared error for the dataset
        """
        squaredError: Optional[Tensor] = None
        seenSamples = 0
        totalBatches = len(data)
        for batchIndex, (features, targets) in enumerate(data):
            if device is not None:
                features = features.to(device)
                targets = targets.to(device)
            # cannot do this outside the loop as we don't know the number of targets yet
            if squaredError is None:
                squaredError = torch.zeros_like(targets, device=device)
            predicted = self.predict(features)
            squaredError += (predicted - targets) ** 2
            seenSamples += targets.shape[0]
            print(f"Evaluating regressor batch {batchIndex + 1}/{totalBatches}", end="\r")
        else:
            # this only happens if we have no data
            squaredError = Tensor([0])
        return (squaredError / seenSamples).cpu()

    def evaluateSplits(self, ds: CsvDatasetSplits) -> None:
        """
        Evaluates the model on the given dataset splits and logs the final MSE
        :param ds: Dataset splits to evaluate
        """
        logging.info(f"MSE for train: {self.evaluateDataset(ds.train)}")
        logging.info(f"MSE for validate: {self.evaluateDataset(ds.validate)}")
        logging.info(f"MSE for test: {self.evaluateDataset(ds.test)}")

    def evaluateDataLoaders(self, train: DataLoader, validate: DataLoader, test: DataLoader, device: torch.device = None
                            ) -> None:
        """
        Evaluates the model on the given dataset splits and logs the final MSE
        :param train:    Loader for training data
        :param validate: Loader for validation data
        :param test:     Loader for testing data
        :param device:   Device to use for computation
        """
        logging.info(f"MSE for train: {self.evaluateDataloader(train, device)}")
        logging.info(f"MSE for validate: {self.evaluateDataloader(validate, device)}")
        logging.info(f"MSE for test: {self.evaluateDataloader(test, device)}")


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
        # TODO: target index as we might have multiple targets
        return self.nn(features)
