from typing import Tuple

import torch
from overrides import override
from torch import Tensor, Generator
from torch.distributions import Dirichlet
from torch.nn import Module

from .decision import DecisionMaker, computeBestActions
from ..dataset.mutators import fullyObservedMask
from ..model.loss import safeNormalize
from ..model.regressor import Regressor, NeuralNetworkRegressor


class DirichletDecisionMaker(DecisionMaker):
    """Decision maker predicting a beta distribution based on `Resnet18Dirichlet`. The model outputs Dirichlet strength values for each parameter."""

    regressor: Regressor
    """Classifier to use to make predictions, should behave like `Resnet18Dirichlet`"""
    missingValue: float
    """Value to assign to nan features"""
    size: torch.Size
    """Samples to take from the distribution"""

    def __init__(self, regressor: Regressor, distSamples: int, scale: float = 1, missingValue: float = 0):
        self.regressor = regressor
        self.size = torch.Size((distSamples,))
        self.scale = scale
        self.missingValue = missingValue

    @override
    def estimateBestAction(self, features: Tensor, lossFunction: callable, actions: Tensor, rand: Generator = None,
                           indices: Tensor = None) -> Tuple[Tensor, Tensor]:
        # replace nan with the missing value; allows mixing dirichlet and non with the different mask formats
        nans = torch.isnan(features)
        if nans.any():
            features = features.clone()
            features[nans] = self.missingValue

        alphas = self.regressor.predict(features)
        assert alphas.shape[0] == features.shape[0]
        # No worry of zero variance, so can directly construct the distribution over the full set of alphas
        distribution = Dirichlet(alphas)

        # phis has dimension (randomSamples, datasetSamples, featureCount), but we want (datasetSamples, randomSamples, featureCount)
        phis = torch.swapaxes(distribution.sample(self.size), 0, 1)
        return computeBestActions(phis, lossFunction, actions)

    @property
    @override
    def name(self):
        return "Dirichlet Network"


class DirichletClassifier(NeuralNetworkRegressor):
    """
    Wrapper around NeuralNetworkRegressor containing `Resnet18Dirichlet` that ensures the results conform to a standard mean predicting classifier instead of strength predicting.
    Allows using it in all method of moment approaches as a baseline for comparison.
    """

    mask_dim: int
    """Dimension containing the mask"""
    expected_mask_size: int
    """Size the mask should be to run the Dirichlet classifier"""


    def __init__(self, nn: Module, num_classes: int, mask_dim: int = 1, expected_mask_size: int = 4):
        super().__init__(nn, safeNormalize, 1 if num_classes == 1 else -1)
        self.mask_dim = mask_dim
        self.expected_mask_size = expected_mask_size

    @classmethod
    def fromRegressor(cls, regressor: Regressor, *args, **kwargs) -> "DirichletClassifier":
        if isinstance(regressor, NeuralNetworkRegressor):
            return cls(regressor.nn, *args, **kwargs)
        raise ValueError("Can only convert a NeuralNetworkRegressor to DirichletRegressor")

    @override
    def predict(self, features: Tensor) -> Tensor:
        # its possible we were given an image with 3 channels. If so, add the mask as all "present"
        maskSize = features.shape[self.mask_dim]
        if maskSize != self.expected_mask_size:
            # need to determine if we wanted 1 mask channel, or 1 per in the original image
            missingSize = self.expected_mask_size - maskSize
            assert missingSize == 1 or missingSize == maskSize, "Mask must either double the size or add 1 to the size"
            features = fullyObservedMask(features, missingSize == 1, dim=self.mask_dim)
            assert features.shape[self.mask_dim] == self.expected_mask_size, "Wrong mask size after adding fully observed mask"
        return super().predict(features)
