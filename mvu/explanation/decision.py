from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import torch
from torch import Tensor, Generator
from torch.distributions import Distribution

from mvu.model.common import CachableModel, Namable


def computeActionConfidence(phi: Tensor, lossFunction: callable, action: int, actions: Tensor) -> float:
    """
    Computes the probability that the given action dominates all other actions.
    :param phi:           Samples from the phi distribution
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param action:        Primary action to consider
    :param actions:       Tensor of all possible actions
    :return:  Probability that action dominates all other actions
    """
    allLoss: Tensor
    if len(phi.shape) == 1:
        # compute loss for each action, outer product makes dimensions [sampleIdx, actionIdx]
        allLoss = torch.outer(phi, lossFunction(1, actions)) + torch.outer(1 - phi, lossFunction(0, actions))
    else:
        allLoss = torch.zeros((phi.shape[0], actions.shape[0]))
        for i in range(phi.shape[1]):
            allLoss += torch.outer(phi[:, i], lossFunction(i, actions))

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
    allLoss: Tensor
    # if the dimension is (datasetSamples,1), convert to (datasetSamples,) for simplicity
    if len(phi.shape) > 1:
        phi = phi.squeeze(1)
    # if the dimension is (), convert to (1,)
    elif len(phi.shape) == 0:
        phi = phi.unsqueeze(0)

    # dimension of phi is (datasetSamples,)
    if len(phi.shape) == 1:
        # compute loss for each action, outer product makes dimensions [sampleIdx, actionIdx]
        allLoss = torch.outer(phi, lossFunction(1, actions)) + torch.outer(1 - phi, lossFunction(0, actions))
    else:
        # dimension of phi is (datasetSamples, featureCount)
        allLoss = torch.zeros((phi.shape[0], actions.shape[0]), dtype=torch.float, device=phi.device)
        for i in range(phi.shape[1]):
            allLoss += torch.outer(phi[:, i], lossFunction(i, actions))

    # determine the number of times each action is the best using bincount
    bestCounts = allLoss.min(axis=1).indices.bincount(minlength=actions.shape[0])
    assert bestCounts.shape[0] == actions.shape[0]
    # find which index is the best overall
    bestIndex = bestCounts.max(dim=0).indices
    assert len(bestIndex.shape) == 0
    # map the index back to an action, and get its probability
    return actions[bestIndex], bestCounts[bestIndex] / bestCounts.sum()


def computeBestActions(phis: Union[List[Tensor], Tensor], lossFunction: callable, actions: Tensor
                       ) -> Tuple[Tensor, Tensor]:
    """
    Computes a tensor of best actions for the given Phi
    :param phis:           Iteratable where the first index is size inputSamples, and the second size distSamples
    :param lossFunction:  Loss function, first parameter is the Y label (0 or 1) and second is the action tensor
    :param actions:       Tensor of all possible actions
    :return: Vector of actions and confidences of size inputSamples
    """
    inputSamples = len(phis)
    device = phis[0].device
    bestActions = torch.empty((inputSamples,), dtype=torch.int, device=device)
    confidences = torch.empty((inputSamples,), dtype=torch.float, device=device)
    for i, phi in enumerate(phis):
        action, confidence = computeBestAction(phi, lossFunction, actions)
        bestActions[i] = action
        confidences[i] = confidence
    return bestActions, confidences


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


class DecisionMaker(CachableModel, Namable, ABC):
    """
    Logic to make a decision given a input tensor (with missingness) and an input action space (with loss function)
    """

    scale: float
    """Calibration scale for this method. If this method does not scale, set it to 1 in the constructor."""

    @abstractmethod
    def estimateBestAction(self, features: Tensor, lossFunction: callable, actions: Tensor, rand: Generator = None,
                           indices: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Estimates the best action for the given features list.
        :param features:      List of features with missingness, size is (inputSamples, featureDim...)
        :param lossFunction:  Loss function to evaluate
        :param actions:       List of valid actions
        :param rand:          Random state
        :param indices:       Sample indices for the sake of caching. This should only be used to reduce computation times,
                              not in any way that provides access to normally hidden data.
        :return:  Tuple of actions (size inputSamples) and action confidences (size inputSamples)
        """
        pass
