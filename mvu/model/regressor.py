import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Optional, Union, List

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

    @staticmethod
    @torch.no_grad()
    def evaluateData(data: DataLoader, batchCallback: callable = None) -> Tensor:
        """
        Evaluates the model on the given dataset and logs the final MSE
        :param data:          Data to evaluate
        :param batchCallback: Runs the loss on the batch. Given batch, returns (batchLoss, batchSamples)
        :return:  Mean squared error for the dataset
        """
        totalLoss = 0
        seenSamples = 0
        totalBatches = len(data)
        startTime = perf_counter()
        for batchIndex, batch, in enumerate(data):
            batchLoss, batchSamples = batchCallback(*batch)
            totalLoss += batchLoss
            seenSamples += batchSamples
            print(f"Evaluating regressor batch {batchIndex + 1}/{totalBatches}", end="\r")
        # this only happens if we have no data
        if totalBatches == 0:
            return torch.tensor([0])
        logging.info(f"Evaluated regressor with data loader in {perf_counter() - startTime:.5f} seconds")
        return torch.tensor([totalLoss / totalBatches])

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

        def batchHandler(features: Tensor, targets: Tensor):
            if device is not None:
                features = features.to(device)
                targets = targets.to(device)
            predicted = self.predict(features)
            # TODO: bring back per feature loss? would need a custom loss function and to ditch the item call here
            # might at that point want multiple loss function support
            loss = lossFunction(predicted, targets).item()
            return loss, targets.shape[0]

        return Regressor.evaluateData(data, batchHandler)

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


def _identity(features: Tensor) -> Tensor:
    """Simple identity function for fallback"""
    return features


class NaiveLinearRegressor(Regressor):
    """
    Simple classifier using a linear combination for weights and a sigmoid to map that to probability values.
    Saves needing to deal with any libraries or setup a neural network for a simple setup.
    """

    weights: Tensor
    """Weights for each feature in the classifier"""
    bias: Tensor
    """Constant offset from the inner product"""
    activation: callable
    """Function mapping a tensor to the output"""

    def __init__(self, weights: Union[Tensor, List[float]], bias: Union[Tensor, List[float], float] = 0,
                 activation: callable = None):
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.bias = torch.as_tensor(bias, device=self.weights.device)
        self.activation = _identity if activation is None else activation

        # validate final parameters
        weightSize = len(self.weights.shape)
        assert weightSize == 1 or weightSize == 2, "Weights must be a vector or matrix"
        outSize = self.weights.shape[0] if weightSize == 2 else 1
        if len(self.bias.shape) > 0:
            assert len(self.bias.shape) == 1 and self.bias.shape[0] == outSize, \
                f"Bias must be a vector of the output size, got {self.bias.shape}, expected {outSize}"

    @override
    def predict(self, features: Tensor) -> Tensor:
        return self.activation(torch.inner(features, self.weights) + self.bias)

    @override
    def to(self, device: torch.device) -> None:
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
