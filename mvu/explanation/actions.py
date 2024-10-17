from typing import Tuple, Optional

import torch
from torch import Tensor


ZERO_ONE_ACTIONS = torch.tensor([0, 1])
"""Zero-one action space, for a boolean input. Use with `zeroOneLoss`"""


def zeroOneLoss(label, action) -> Tensor:
    """
    Basic zero-one loss function for a set of actions. Designed for binary but should work on larger spaces.
    :param label:
    :param action:
    :return:
    """
    return torch.as_tensor(label != action, dtype=torch.float)


ALEATORIC_ACTIONS = torch.tensor([0, 1, -1])
"""Aleatoric action space, for a boolean input. Use with `createAleatoricLoss`"""


def createAleatoricLoss(constantLoss: float = 0.25) -> callable:
    """
    Creates a loss function for a zero-one loss with a constant loss -1 action for aleatoric uncertainty.
    Designed for binary but should work on larger spaces.
    :param constantLoss:  Constant aleatoric action loss
    :return: Loss function callable
    """

    def loss(label, action) -> Tensor:
        action = torch.as_tensor(action)
        # zero one loss, but a label of -1 has reduced penalty
        # assuming label is never -1
        return torch.ne(action, label).float() - (torch.eq(action, -1).float() * (1 - constantLoss))
    return loss


def createActionSpace(name: str, size: int, constantLoss: float = 0.25, device: Optional[torch.device] = None
                      ) -> Tuple[callable, Tensor]:
    """
    Creates an action space by name
    :param name:           Action space name
    :param size:           Number of features in the action space
    :param constantLoss:   Value for constant loss if using aleatoric space
    :param device:         Device for evaluations
    :return:  Pair of loss function and action tensor
    """
    if name == "zero-one":
        if size == 1:
            actions = torch.tensor((0, 1), dtype=torch.int, device=device)
        else:
            actions = torch.arange(1, size+1, dtype=torch.int, device=device)
        return zeroOneLoss, actions
    if "aleatoric" in name:
        if size == 1:
            actions = torch.tensor((0, 1, -1), dtype=torch.int, device=device)
        else:
            actions = torch.arange(1, size+2, dtype=torch.int, device=device)
            actions[size] = -1
        return createAleatoricLoss(constantLoss), actions
    else:
        raise ValueError(f"Unknown mask name '{name}'")
