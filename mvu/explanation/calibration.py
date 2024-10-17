import logging
from time import perf_counter
from typing import Optional

import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .decision import DecisionMaker, computeBestActions
from ..logger import handleException
from ..model.regressor import Regressor


def bestActionWithoutMissing(features: Tensor, classifier: Regressor, lossFunction: callable, actions: Tensor
                             ) -> Tensor:
    assert torch.count_nonzero(torch.isnan(features)) == 0, "Cannot compute baseline best action with missingness"
    phis = classifier.predict(features)
    actions, confidences = computeBestActions(phis.reshape(-1, 1), lossFunction, actions)
    return actions


def computeMVCE(cleanLoader: DataLoader, mutatedLoader: DataLoader, decisionMaker: DecisionMaker,
                lossFunction: callable, actions: Tensor, buckets: int,
                classifier: Regressor = None, rand: Generator = None, device: Optional[torch.device] = None) -> Tensor:
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
    :param device:          Device to use for calculations
    :return:  Computed missing value calibration error
    """
    bucketSizes = torch.zeros((buckets,), dtype=torch.int, device=device)
    bucketConfidence = torch.zeros((buckets,), dtype=torch.float, device=device)
    bucketConsistency = torch.zeros((buckets,), dtype=torch.float, device=device)
    time = perf_counter()

    for i, (cleanBatch, mutatedBatch) in enumerate(zip(cleanLoader, mutatedLoader)):
        mutatedFeatures = mutatedBatch[0]
        sampleIndices: Optional[Tensor] = None
        supportedIndices: Tensor
        # if we have indices, ensure they match then pass them along
        if len(mutatedBatch) == 3:
            sampleIndices = mutatedBatch[2]
            # ensure that all samples in this batch can be processed, lets us skip non-cached images when using a cache
            supportedIndices = decisionMaker.supportsIndices(sampleIndices)
            # skip the batch if it has no processable samples
            if torch.count_nonzero(supportedIndices) == 0:
                continue
            sampleIndices = sampleIndices[supportedIndices]
        else:
            supportedIndices = torch.ones((mutatedFeatures.shape[0],), dtype=torch.bool)
        mutatedFeatures = mutatedFeatures[supportedIndices]

        # if the classifier is None, it means our dataloader contains the best actions
        bestActions: Tensor
        if classifier is None:
            bestActions = cleanBatch[mutatedFeatures]
            assert len(bestActions.shape) == 1, "Best actions batch must be a vector"
            assert bestActions.shape[0] == mutatedFeatures.shape[0], \
                "Clean and mutated dataset must have the same batch size"
            if device is not None:
                bestActions = bestActions.to(device)
        else:
            # if we have a classifier, the batch is (features, labels, [indices])
            # enforce index match
            if len(cleanBatch) == 3:
                assert torch.count_nonzero(torch.ne(sampleIndices, cleanBatch[2][supportedIndices])) == 0, \
                    f"Received batch {i} of data with mismatching cache indices, likely invalid datasets"
            # compute best actions with respect to clean data
            cleanFeatures = cleanBatch[0][supportedIndices]
            if device is not None:
                cleanFeatures = cleanFeatures.to(device)
            assert cleanFeatures.shape == mutatedFeatures.shape, \
                "Clean and mutated dataset must have the same batch shape"
            bestActions = bestActionWithoutMissing(cleanFeatures, classifier, lossFunction, actions)

        # compute predicted actions
        if device is not None:
            mutatedFeatures = mutatedFeatures.to(device)
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

    totalSamples = bucketSizes.sum()
    mvce = (bucketSizes * torch.abs(bucketConsistency - bucketConfidence)).sum() / totalSamples
    time = perf_counter() - time
    logging.info(f"""
        Computed MVCE {mvce.cpu().item()} for {decisionMaker.name} in {time} seconds with {buckets} buckets:
        * Non-zero buckets: {torch.nonzero(nonZero).squeeze().cpu()}
        * Final bucket sizes: {bucketSizes.cpu()} totaling {totalSamples.cpu().item()} samples
        * Final bucket confidences: {bucketConfidence.cpu()}
        * Final bucket consistencies: {bucketConsistency.cpu()}
    """)

    # compute final MVCE metric
    return mvce


class CalibrationExperiment:
    # mvce parameters, see `computeMVCE` for docs
    cleanLoader: DataLoader
    mutatedLoader: DataLoader
    decisionMaker: DecisionMaker
    lossFunction: callable
    actions: Tensor
    buckets: int
    classifier: Regressor
    rand: Generator
    device: torch.device

    # additional parameters
    maskName: str
    """Name of the missing region"""
    actionName: str
    """Name of the action space"""
    trials: int
    """Number of times to compute the MVCE, for the sake of error bars"""
    time: Optional[float]
    """Duration of this experiment"""
    results: Tensor
    """MVCE results for this experiment, size is equal to trials"""

    def __init__(self, cleanLoader: DataLoader, maskName: str, mutatedLoader: DataLoader, decisionMaker: DecisionMaker,
                 actionName: str, lossFunction: callable, actions: Tensor, buckets: int, trials: int,
                 classifier: Regressor = None, rand: Generator = None, device: Optional[torch.device] = None):
        self.cleanLoader = cleanLoader
        self.maskName = maskName
        self.mutatedLoader = mutatedLoader
        self.decisionMaker = decisionMaker

        self.actionName = actionName
        self.lossFunction = lossFunction
        self.actions = actions

        self.buckets = buckets
        self.classifier = classifier
        self.rand = rand
        self.device = device
        self.trials = trials
        self.time = None

    @property
    def experimentName(self):
        """Name of the overall experiment"""
        return f"{self.decisionMaker.name} missing {self.maskName} in {self.actionName}"

    def __call__(self, *args, **kwargs):
        logging.info(f"Started running {self.experimentName}")
        startTime = perf_counter()

        try:
            self.results = torch.empty((self.trials,), dtype=torch.float)
            for i in range(self.trials):
                self.results[i] = computeMVCE(
                    self.cleanLoader, self.mutatedLoader, self.decisionMaker,
                    self.lossFunction, self.actions, self.buckets, self.classifier,
                    self.rand, self.device
                ).cpu()
        except KeyboardInterrupt as e:
            # this is just logging the context so we know which experiment was terminated
            # its in the log again later and earlier, but this reduces some of the debug time
            logging.error(f"Received keyboard interrupt during {self.experimentName}, terminating program")
            raise e
        except BaseException as e:
            handleException(type(e), e, e.__traceback__,
                            message=f"Failed to process {self.experimentName}")

        # store final experiment time
        endTime = perf_counter()
        self.time = endTime - startTime
        logging.info(f"Finished running {self.experimentName} in {self.time} seconds")

    @classmethod
    def writeResultHeaders(cls, csvFile, trials: int) -> None:
        """
        Writes the result header to the file
        :param csvFile:  CSV file for result writing
        :param trials:   Number of trial headers to include
        """
        csvFile.writerow([
            "Method", "Action Space", "Mask", "Time",
            "MVCE Mean", "MVCE Std",
            *[f"Trial {i+1}" for i in range(trials)]
        ])

    def writeResults(self, csvFile) -> None:
        """
        Writes the results to the file
        :param csvFile:  CSV file for result writing
        """
        if self.time is None:
            logging.error(f"Skipping including {self.experimentName} in result CSV as it did not complete.")

        csvFile.writerow([
            self.decisionMaker.name, self.actionName, self.maskName, self.time,
            self.results.mean().item(), self.results.std().item(),
            *[result.item() for result in self.results]
        ])
