from typing import Tuple

import torch
from overrides import override
from torch import Tensor, Generator
from torch.distributions import Dirichlet

from .decision import DecisionMaker, computeBestActions
from ..model.regressor import Regressor


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

        # phis has dimension (randomSamples, datasetSamples, featureCount)
        phis = distribution.sample(self.size)
        return computeBestActions(phis, lossFunction, actions)

    @property
    @override
    def name(self):
        return "Dirichlet Network"
