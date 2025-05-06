from typing import Type, Tuple

import torch
from overrides import override
from torch import Tensor
from torch.distributions import Distribution, Dirichlet
from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import sigmoid, normalize


class CrossEntropyProbabilityLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        return super().forward(torch.log(probs), target)


class DistributionLoss(Module):
    """
    Loss function that includes a log probability value alongside a loss function on the clean image
    """

    probLoss: Module
    """Loss function to run on the probability vector"""
    distClass: Type[Distribution]
    """Distribution constructor for the masked image loss"""
    maskedWeight: float
    """Weight to apply to the masked probability loss"""
    distWeight: float
    """Weight to apply to the distribution loss"""
    cleanWeight: float
    """Weight to apply to the clean probability loss"""

    def __init__(self, probLoss: Module, distClass: Type[Distribution],
                 maskedWeight: float, distWeight: float, cleanWeight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.probLoss = probLoss
        self.distClass = distClass
        self.maskedWeight = maskedWeight
        self.distWeight = distWeight
        self.cleanWeight = cleanWeight
        self.reduction = reduction

    def _toProbability(self, cleanResult: Tensor):
        return cleanResult

    def _forward(self, cleanResult: Tensor, maskedResult: Tensor, maskedParameters: Tuple[Tensor], target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:          Result from the clean image
        :param maskedResult:         Result from the masked image
        :param maskedParameters:     Parameters from the masked image
        :param target:         Target class from the dataset
        :return:  Combined loss
        """
        loss = 0
        if self.cleanWeight > 0:
            loss += self.probLoss(cleanResult, target) * self.cleanWeight
        if self.maskedWeight > 0:
            loss += self.probLoss(maskedResult, target) * self.maskedWeight
        if self.distWeight > 0:
            distLoss = -self.distClass(*maskedParameters).log_prob(normalize(self._toProbability(cleanResult), p=1))
            if self.reduction == "mean":
                distLoss = distLoss.mean()
            elif self.reduction == "sum":
                distLoss = distLoss.sum()
            loss += distLoss * self.distWeight
        return loss


class DirichletLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution.

    Typically used with BCELoss.
    """
    def __init__(self, maskedWeight: float, distWeight: float, cleanWeight: float = 1.0,
                 cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(cleanLoss if cleanLoss is not None else CrossEntropyProbabilityLoss(), Dirichlet,
                         maskedWeight, distWeight, cleanWeight, reduction)

    def forward(self, cleanResult: Tensor, maskedResult: Tensor, target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as strengths
        :param maskedResult:  Result of the masked image through the neural network as strengths
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        return self._forward(
            normalize(cleanResult, p=1),
            normalize(maskedResult, p=1),
            (maskedResult,),
            target
        )


class DirichletStrengthLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution.

    Typically used with BCELoss.
    """
    def __init__(self, cleanLoss: Module, maskedWeight: float, distWeight: float, cleanWeight: float = 1.0,
                 reduction: str = "mean"):
        super().__init__(cleanLoss, Dirichlet, maskedWeight, distWeight, cleanWeight, reduction)

    def forward(self, cleanResult: Tuple[Tensor], maskedResult: Tuple[Tensor], target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as (probabilities, strength)
        :param maskedResult:  Result of the masked image through the neural network as (probabilities, strength)
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        return self._forward(
            cleanResult[0],
            maskedResult[0],
            (self._toProbability(maskedResult[0]) * maskedResult[1].unsqueeze(1),),
            target
        )


class DirichletStrengthLogitLoss(DirichletStrengthLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution with output logits.

    Typically used with CrossEntropyLoss, but BCEWithLogitsLoss also works.
    """
    def __init__(self, maskedWeight: float, distWeight: float, cleanWeight: float = 1.0,
                 cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(cleanLoss if cleanLoss is not None else CrossEntropyLoss(),
                         maskedWeight, distWeight, cleanWeight, reduction)

    @override
    def _toProbability(self, cleanResult: Tensor):
        return torch.exp(cleanResult)
