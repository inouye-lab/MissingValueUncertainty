import torch
from torch import Tensor
from torch.distributions import Distribution


def computeProbabilityActionDominated(phi: Tensor, lossFunction: callable, action: int, actions: Tensor) -> float:
    """
    Computes the probability that the given action is dominated by another action.
    :param phi:           Samples from the phi distribution
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action is dominated by another action
    """
    # compute loss for each action, outer product makes dimensions [sampleIdx, actionIdx]
    allLoss = torch.outer(phi, lossFunction(1, actions)) + torch.outer(1 - phi, lossFunction(0, actions))
    # split loss into term for the main action and for all other actions
    # TODO: I feel like there should be a way to create an index tensor of "actions in set" instead of "action == value"
    # if we do that, actionLoss should also be a min expression instead of a squeeze
    actionLoss = allLoss[:, actions == action].squeeze()
    otherLoss = allLoss[:, actions != action].min(axis=1).values
    # count the number of times this action is dominated as the final output
    return torch.as_tensor(actionLoss > otherLoss, dtype=torch.float).mean().item()


def sampleProbabilityActionDominated(dist: Distribution, samples: int, lossFunction: callable, action: int,
                                     actions: Tensor) -> float:
    """
    Samples the probability that the given action is dominated by another action.
    :param dist:          Distribution to sample
    :param samples:       Number of samples to take
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action is dominated by another action
    """
    return computeProbabilityActionDominated(dist.sample(torch.Size((samples,))), lossFunction, action, actions)