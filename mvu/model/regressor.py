import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Optional

import torch
from overrides import override
from sklearn.linear_model import Ridge
from torch import Tensor
from torch.nn import Module, MSELoss
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
    def evaluateDataloader(self, data: DataLoader, device: torch.device = None, lossFunction: callable = None
                           ) -> Tensor:
        """
        Evaluates the model on the given dataset and logs the final MSE
        :param data:   Data to evaluate
        :param device: Device to use for computation
        :param lossFunction: Loss function, taking parameters of prediction and targets
        :return:  Mean squared error for the dataset
        """
        if lossFunction is None:
            lossFunction = MSELoss()
        totalLoss = 0
        seenSamples = 0
        totalBatches = len(data)
        startTime = perf_counter()
        for batchIndex, (features, targets) in enumerate(data):
            if device is not None:
                features = features.to(device)
                targets = targets.to(device)
            predicted = self.predict(features)
            # TODO: bring back per feature loss? would need a custom loss function and to ditch the item call here
            # might at that point want multiple loss function support
            totalLoss += lossFunction(predicted, targets).item()
            seenSamples += targets.shape[0]
            print(f"Evaluating regressor batch {batchIndex + 1}/{totalBatches}", end="\r")
        # this only happens if we have no data
        if totalBatches == 0:
            return torch.tensor([0])
        logging.info(f"Evaluated regressor with data loader in {perf_counter() - startTime:.5f} seconds")
        return torch.tensor([totalLoss / totalBatches])

    def evaluateSplits(self, ds: CsvDatasetSplits) -> None:
        """
        Evaluates the model on the given dataset splits and logs the final MSE
        :param ds: Dataset splits to evaluate
        """
        logging.info(f"MSE for train: {self.evaluateDataset(ds.train)}")
        logging.info(f"MSE for validate: {self.evaluateDataset(ds.validate)}")
        logging.info(f"MSE for test: {self.evaluateDataset(ds.test)}")

    def evaluateDataLoaders(self, train: DataLoader, validate: DataLoader, test: DataLoader,
                            device: torch.device = None, lossFunction: callable = None, label: str = "MSE") -> None:
        """
        Evaluates the model on the given dataset splits and logs the final MSE
        :param train:    Loader for training data
        :param validate: Loader for validation data
        :param test:     Loader for testing data
        :param lossFunction: Loss function, taking parameters of prediction and targets
        :param label:    Label for output printing
        :param device:   Device to use for computation
        """
        if lossFunction is None:
            lossFunction = MSELoss()
        for (name, loader) in [("train", train), ("validate", validate), ("test", test)]:
            result = self.evaluateDataloader(loader, device, lossFunction)
            logging.info(f"{label} for {name} is {result.mean().item()}:\n{result}")

    def setFeatureIndex(self, featureIndex: int):
        """
        Sets the feature index for this regressor, may be unused.
        @param featureIndex:  Feature index to use, -1 means all features.
        """
        pass

    @abstractmethod
    def to(self, device: torch.device):
        """Sets configuration for this regressor at evaluation"""
        pass


class RidgeRegressor(Regressor):
    """Regressor implemented using the SKLearn Ridge Regression functionality"""

    ridge: Ridge
    device: Optional[torch.device]

    def __init__(self, ridge: Ridge):
        self.ridge = ridge
        self.device = None

    @override
    def predict(self, features: Tensor) -> Tensor:
        return torch.tensor(self.ridge.predict(features.detach().cpu().numpy()), device=self.device)

    @override
    def to(self, device: torch.device):
        self.device = device
        if device.type != 'cpu':
            logging.warning(f"RidgeRegressor does not benefit from device {device}")


class NeuralNetworkRegressor(Regressor):
    """Regressor using a torch neural network"""

    nn: Module
    featureIndex: int

    def __init__(self, nn: Module):
        self.nn = nn
        self.featureIndex = -1

    @override
    def predict(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            return self.predictWithGradient(features)

    def predictWithGradient(self, features: Tensor) -> Tensor:
        """
        Makes a prediction using this regressor and computing gradients.
        :param features: Input tensor, dimension 0 is samples and dimension 1 is features
        :return: Output tensor, with a single dimension representing the prediction
        """
        result = self.nn(features)
        if self.featureIndex > -1:
            return result[:, self.featureIndex]
        return result

    @override
    def setFeatureIndex(self, featureIndex: int):
        self.featureIndex = featureIndex

    @override
    def to(self, device: torch.device):
        self.nn.to(device)
