from typing import Type, Tuple, Optional

import torch
from overrides import override
from torch import Tensor
from torch.distributions import Distribution, Dirichlet
from torch.nn import Module, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import normalize, softmax, sigmoid


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

    def _forward(self, cleanResult: Tensor, maskedResult: Tensor, maskedParameters: Tuple[Tensor], cleanProbability: Tensor, target: Tensor):
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
            # add in a small value to the probability to prevent infinities on perfect match
            distLoss = -self.distClass(*maskedParameters).log_prob(cleanProbability + 1e-35)
            if self.reduction == "mean":
                distLoss = distLoss.mean()
            elif self.reduction == "sum":
                distLoss = distLoss.sum()
            loss += distLoss * self.distWeight
        return loss


def safeNormalize(tensor: Tensor) -> Tensor:
    """Normalizes a vector without issue if the vector contains 0 or infinity"""
    # if a row is all 0s, set them all to 1 before normalizing
    tensor[torch.all(tensor == 0, dim=1),:] = 1
    # if it has infinity, clear then row, but then replace the infinities with 1s
    inf = torch.isinf(tensor)
    tensor[torch.any(inf, dim=1),:] = 0
    tensor[inf] = 1
    # normalize the remainder
    return normalize(tensor, p=1)


class DirichletLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a Dirichlet distribution.

    Typically used with CrossEntropyProbabilityLoss.
    """
    def __init__(self, maskedWeight: float, distWeight: float, cleanWeight: float = 1.0,
                 cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(cleanLoss if cleanLoss is not None else CrossEntropyProbabilityLoss(), Dirichlet,
                         maskedWeight, distWeight, cleanWeight, reduction)

    def forward(self, cleanResult: Tensor, maskedResult: Tensor, target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as alpha values or probabilities
        :param maskedResult:  Result of the masked image through the neural network as alpha values
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        # with just 1 value, our loss function likely expects size 1 tensors
        # but the Dirchlet distribution wants size 2 tensors, so convert as needed
        maskedProbability = safeNormalize(maskedResult.clone()) if self.maskedWeight > 0 else None
        cleanProbability: Tensor
        if cleanResult.shape[1] == 1:
            cleanProbability = cleanResult
            cleanResult = torch.cat((1 - cleanResult, cleanResult), dim=1)
            if self.maskedWeight > 0:
                maskedProbability = maskedProbability[:,1].unsqueeze(dim=1)
        else:
            cleanResult = safeNormalize(cleanResult)
            cleanProbability = cleanResult
        # normalize to ensure vector inputs are probability values
        return self._forward(
            cleanProbability,
            maskedProbability,
            (maskedResult,),
            cleanResult,
            target
        )


class DirichletLogitLoss(DistributionLoss):
    """
    Implementation of `DistributionLoss` for a model outputting logits of a dirichlet distribution.

    Typically used with CrossEntropyLoss.
    """
    def __init__(self, maskedWeight: float, distWeight: float, cleanWeight: float = 1.0,
                 cleanLoss: Module = None, reduction: str = "mean"):
        super().__init__(cleanLoss if cleanLoss is not None else CrossEntropyLoss(), Dirichlet,
                         maskedWeight, distWeight, cleanWeight, reduction)

    def forward(self, cleanResult: Tensor, maskedResult: Tensor, target: Tensor):
        """
        Runs the forward pass for this loss function
        :param cleanResult:   Result of the clean image through the neural network as logits
        :param maskedResult:  Result of the masked image through the neural network as logits
        :param target:        Target class from the dataset
        :return:  Combined loss
        """
        cleanProbability: Tensor
        # clean result should be probabilities, so run sigmoid/softmax, though can't do softmax with just 1 feature
        if cleanResult.shape[1] == 1:
            cleanProbability = sigmoid(cleanResult)
            cleanProbability = torch.cat((1 - cleanProbability, cleanProbability), dim=1)
        else:
            cleanProbability = softmax(cleanResult, dim=1)

        return self._forward(
            cleanResult,
            maskedResult,
            # parameters are expected to be alpha values, so need to run the exp activation
            (torch.exp(maskedResult),),
            cleanProbability,
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

    def _toProbability(self, cleanResult: Tensor):
        return cleanResult

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
            self._toProbability(cleanResult[0]),
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

def createLoss(name: Optional[str], classification: bool = True) -> callable:
    """
    Creates a loss function from command line parameters.
    :param name:           Name of the loss function
    :param classification: Uses classification loss when relevant, notably in defaults.
    """
    if name is None:
        return BCELoss() if classification else MSELoss()
    elif name == "BCE" or name == "BCEProbabilities":
        return BCELoss()
    elif name == "BCELogits":
        return BCEWithLogitsLoss()
    elif name == "MSE":
        return MSELoss()
    elif name == "CEProbabilities":
        return CrossEntropyProbabilityLoss()
    elif name == "CrossEntropy" or name == "CELogits":
        return CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function '{name}'")