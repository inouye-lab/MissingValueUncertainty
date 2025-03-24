from typing import Type, Tuple

import torch
from overrides import override
from torch import Tensor
from torch.distributions import Distribution, Dirichlet
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import sigmoid, normalize


class CrossEntropyProbabilityLoss(CrossEntropyLoss):
    def __init__(self, *args, reduction='sum', **kwargs):
        super().__init__(*args, reduction=reduction, **kwargs)

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        return super().forward(torch.log(probs), target)



class DistributionLoss(Module):
    """
    Loss function that includes a log probability value alongside a loss function on the clean image
    """

    cleanLoss: Module
    """Loss function to run on the clean probability vector"""
    distClass: Type[Distribution]
    """Distribution constructor for the masked image loss"""
    distWeight: float
    """Weight to apply to the distribution loss"""

    def __init__(self, cleanLoss: Module, distClass: Type[Distribution], distWeight: float, reduction: str = "mean"):
        super().__init__()
        self.cleanLoss = cleanLoss
        self.distClass = distClass
        self.distWeight = distWeight
        self.reduction = reduction

    def _toProbability(self, cleanResult: Tensor):
        return cleanResult

    def forward(self, cleanResult: Tensor, maskedParameters: Tuple[Tensor], target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:          Result from the clean image
        :param maskedParameters:     Parameters from the masked image
        :param target:         Target class from the dataset
        :return:  Combined loss
        """
        cleanLoss = self.cleanLoss(cleanResult, target)
        distLoss = -self.distClass(*maskedParameters).log_prob(normalize(self._toProbability(cleanResult), p=1))
        if self.reduction == "mean":
            distLoss = distLoss.mean()
        elif self.reduction == "sum":
            distLoss = distLoss.sum()
        return cleanLoss + self.distWeight * distLoss


class DirichletLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution.

    Typically used with BCELoss.
    """
    def __init__(self, distWeight: float, cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(cleanLoss if cleanLoss is not None else CrossEntropyProbabilityLoss(), Dirichlet, distWeight, reduction)

    def forward(self, cleanResult: Tensor, maskedResult: Tensor, target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as strengths
        :param maskedResult:  Result of the masked image through the neural network as strengths
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        return super().forward(normalize(cleanResult, p=1), (maskedResult,), target)


class DirichletStrengthLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution.

    Typically used with BCELoss.
    """
    def __init__(self, distWeight: float, cleanLoss: Module, reduction: str = "mean"):
        super().__init__(cleanLoss, Dirichlet, distWeight, reduction)

    def forward(self, cleanResult: Tuple[Tensor], maskedResult: Tuple[Tensor], target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as probabilities
        :param maskedResult:  Result of the masked image through the neural network as (probabilities, strength)
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        return super().forward(cleanResult[0], (self._toProbability(maskedResult[0]) * maskedResult[1].unsqueeze(1),), target)


class DirichletStrengthLogitLoss(DirichletStrengthLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution with output logits.

    Typically used with CrossEntropyLoss, but BCEWithLogitsLoss also works.
    """
    def __init__(self, distWeight: float, cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(distWeight, cleanLoss if cleanLoss is not None else CrossEntropyLoss(), reduction)

    @override
    def _toProbability(self, cleanResult: Tensor):
        return sigmoid(cleanResult)
