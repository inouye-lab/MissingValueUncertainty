from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution


def computeActionConfidence(phi: Tensor, lossFunction: callable, action: int, actions: Tensor) -> float:
    """
    Computes the probability that the given action dominates all other actions.
    :param phi:           Samples from the phi distribution
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action dominates all other actions
    """
    # compute loss for each action, outer product makes dimensions [sampleIdx, actionIdx]
    allLoss = torch.outer(phi, lossFunction(1, actions)) + torch.outer(1 - phi, lossFunction(0, actions))
    # split loss into term for the main action and for all other actions
    # TODO: I feel like there should be a way to create an index tensor of "actions in set" instead of "action == value"
    # if we do that, actionLoss should also be a min expression instead of a squeeze
    actionLoss = allLoss[:, actions == action].squeeze()
    # TODO: we could simplify this to an arg min, right? average over to find confidence of best action
    otherLoss = allLoss[:, actions != action].min(axis=1).values
    # count the number of times this action is dominated as the final output
    return torch.as_tensor(actionLoss <= otherLoss, dtype=torch.float).mean().item()


def sampleActionConfidence(dist: Distribution, samples: int, lossFunction: callable, action: int, actions: Tensor
                           ) -> float:
    """
    Samples the probability that the given action dominates all other actions.
    :param dist:          Distribution to sample
    :param samples:       Number of samples to take
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action dominates all other actions
    """
    return computeActionConfidence(dist.sample(torch.Size((samples,))), lossFunction, action, actions)


def computeBestAction(phi: Tensor, lossFunction: callable, actions: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Determines the best action for the given phi samples, actions, and loss function.
    :param phi:           Samples from the phi distribution
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param actions:       Tensor of all possible actions
    :return:  Best action tensor, and the confidence tensor of that action
    """
    # compute loss for each action, outer product makes dimensions [sampleIdx, actionIdx]
    allLoss = torch.outer(phi, lossFunction(1, actions)) + torch.outer(1 - phi, lossFunction(0, actions))
    # determine the number of times each action is the best using bincount
    bestCounts = allLoss.min(axis=1).indices.bincount(minlength=actions.shape[0])
    assert bestCounts.shape[0] == actions.shape[0]
    # find which index is the best overall
    bestIndex = bestCounts.max(dim=0).indices
    assert len(bestIndex.shape) == 0
    # map the index back to an action, and get its probability
    return actions[bestIndex], bestCounts[bestIndex] / bestCounts.sum()


def sampleBestAction(dist: Distribution, samples: int, lossFunction: callable, actions: Tensor
                     ) -> Tuple[Tensor, Tensor]:
    return computeBestAction(dist.sample(torch.Size((samples,))), lossFunction, actions)


def computeProbabilityActionDominated(phi: Tensor, lossFunction: callable, action: int, actions: Tensor) -> float:
    """
    Computes the probability that the given action is dominated by another action.

    **Deprecated**: use `computeActionConfidence` instead.
    :param phi:           Samples from the phi distribution
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action is dominated by another action
    """
    return 1 - computeActionConfidence(phi, lossFunction, action, actions)


def sampleProbabilityActionDominated(dist: Distribution, samples: int, lossFunction: callable, action: int,
                                     actions: Tensor) -> float:
    """
    Samples the probability that the given action is dominated by another action.

    **Deprecated**: use `sampleActionConfidence` instead.
    :param dist:          Distribution to sample
    :param samples:       Number of samples to take
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action is dominated by another action
    """
    return computeProbabilityActionDominated(dist.sample(torch.Size((samples,))), lossFunction, action, actions)
