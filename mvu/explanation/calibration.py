import logging
from time import perf_counter
from typing import Optional

import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .decision import DecisionMaker, computeBestActions
from ..model.regressor import Regressor


def bestActionWithoutMissing(features: Tensor, classifier: Regressor, lossFunction: callable, actions: Tensor
                             ) -> Tensor:
    assert torch.count_nonzero(torch.isnan(features)) == 0, "Cannot compute baseline best action with missingness"
    phis = classifier.predict(features)
    actions, confidences = computeBestActions(phis.reshape(-1, 1), lossFunction, actions)
    return actions


def computeMVCE(cleanLoader: DataLoader, mutatedLoader: DataLoader, decisionMaker: DecisionMaker,
                lossFunction: callable, actions: Tensor, buckets: int,
                classifier: Regressor = None, rand: Generator = None) -> Tensor:
    """
    Computes the missing value calibration error for the given decision maker.
    :param cleanLoader:     DataLoader for data with no missingness,
                            or dataloader for best actions if classifer is None.
    :param mutatedLoader:   DataLoader for data with missingness.
    :param decisionMaker:   Logic to make a decision given a feature tensor.
    :param lossFunction:    Loss function for the action space.
    :param actions:         Action space.
    :param buckets:         Number of buckets for computing the calibration error.
    :param classifier:      Classifier for predicting best actions without missingness.
                            If none, cleanLoader is assumed best actions.
    :param rand:            Random state.
    :return:  Computed missing value calibration error
    """
    bucketSizes = torch.zeros((buckets,), dtype=torch.int)
    bucketConfidence = torch.zeros((buckets,), dtype=torch.float)
    bucketConsistency = torch.zeros((buckets,), dtype=torch.float)
    time = perf_counter()

    for i, (cleanBatch, mutatedBatch) in enumerate(zip(cleanLoader, mutatedLoader)):
        sampleIndices: Optional[Tensor] = None
        # if we have indices, ensure they match then pass them along
        if len(mutatedBatch) == 3:
            sampleIndices = mutatedBatch[2]
        mutatedFeatures = mutatedBatch[0]

        # if the classifier is None, it means our dataloader contains the best actions
        bestActions: Tensor
        if classifier is None:
            bestActions = cleanBatch
            assert bestActions.shape[0] == mutatedFeatures.shape[0], \
                "Clean and mutated dataset must have the same batch size"
        else:
            # if we have a classifier, the batch is (features, labels, [indices])
            # enforce index match
            if len(cleanBatch) == 3:
                assert torch.count_nonzero(torch.ne(sampleIndices, cleanBatch[2])) == 0, \
                    f"Received batch {i} of data with mismatching cache indices, likely invalid datasets"
            # compute best actions with respect to clean data
            cleanFeatures = cleanBatch[0]
            assert cleanFeatures.shape == mutatedFeatures.shape, \
                "Clean and mutated dataset must have the same batch shape"
            bestActions = bestActionWithoutMissing(cleanFeatures, classifier, lossFunction, actions)

        # compute predicted actions
        predActions, confidences = decisionMaker.estimateBestAction(
            mutatedFeatures, lossFunction, actions, rand=rand, indices=sampleIndices
        )

        # map the confidence values to the bucket index
        bucketIndices = (confidences * buckets).int()
        # any confidence of 1.0 gets mapped to max bucket
        bucketIndices[bucketIndices == buckets] = buckets - 1

        # consistency metric: actions match best actions
        consistency = torch.eq(predActions, bestActions)

        # bucketSizes = indices.bincount(minlength = buckets)
        for bucket in range(buckets):
            bucketMask = bucketIndices == bucket
            bucketSize = torch.count_nonzero(bucketMask)
            bucketSizes[bucket] += bucketSize
            if bucketSize > 0:
                bucketConfidence[bucket] += confidences[bucketMask].sum()
                bucketConsistency[bucket] += torch.count_nonzero(consistency[bucketMask])

    # up until now, bucketConfidence and bucketConsistency have been sums, need to divide by total size for prob
    # need to be careful about divide by zero though, so skip empty buckets
    nonZero = torch.ne(bucketSizes, 0)
    bucketSizes = bucketSizes[nonZero]
    bucketConfidence = bucketConfidence[nonZero] / bucketSizes
    bucketConsistency = bucketConsistency[nonZero] / bucketSizes

    mvce = (bucketSizes * torch.abs(bucketConsistency - bucketConfidence)).sum() / bucketSizes.sum()
    time = perf_counter() - time
    logging.info(f"Computed MVCE {mvce.item()} in {time} seconds with {buckets} buckets:")
    logging.info(f"Non-zero buckets: {torch.nonzero(nonZero).squeeze()}")
    logging.info(f"Final bucket sizes: {bucketSizes}")
    logging.info(f"Final bucket confidences: {bucketConfidence}")
    logging.info(f"Final bucket consistencies: {bucketConsistency}")

    # compute final MVCE metric
    return mvce
