import logging
from time import perf_counter
from typing import Optional, List, Union, Tuple, NamedTuple

import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .decision import DecisionMaker, computeBestActions
from ..logger import handleException
from ..model.regressor import Regressor


class SoftHardTensor(NamedTuple):
    """Pair of soft confidence and hard confidence versions of the same tensor"""

    soft: Tensor
    """Result from the soft prediction"""
    hard: Tensor
    """Result from the hard prediction"""


def bestActionWithoutMissing(features: Tensor, classifier: Regressor, lossFunction: callable, actions: Tensor
                             ) -> Tensor:
    assert torch.count_nonzero(torch.isnan(features)) == 0, "Cannot compute baseline best action with missingness"
    phis = classifier.predict(features)
    # unsqueeze 1 is for the 1 sample of phi (instead of many)
    actions, confidences = computeBestActions(phis.unsqueeze(1), lossFunction, actions)
    return actions


class MVCEResults(NamedTuple):
    """Results returned from `computeMVCE`"""

    mvce: SoftHardTensor
    """Final MVCE metric score, primary return"""
    confidence: Tensor
    """Average confidence for all samples"""
    accuracy: SoftHardTensor
    """Pair of accuracy percentage using the soft and hard predictions"""
    consistency: SoftHardTensor
    """Pair of consistency percentages using the soft and hard predictions"""


def computeMVCE(cleanLoader: DataLoader, mutatedLoader: DataLoader, decisionMaker: DecisionMaker,
                lossFunction: Union[callable,List[callable]], actions: Tensor, buckets: int,
                classifier: Regressor = None, rand: Generator = None, device: Optional[torch.device] = None
                ) -> MVCEResults:
    """
    Computes the missing value calibration error for the given decision maker.
    :param cleanLoader:     DataLoader for data with no missingness,
                            or dataloader for best actions if classifer is None.
    :param mutatedLoader:   DataLoader for data with missingness.
    :param decisionMaker:   Logic to make a decision given a feature tensor.
    :param lossFunction:    Loss function for the action space. May be a list to perform an expectation over all options
    :param actions:         Action space.
    :param buckets:         Number of buckets for computing the calibration error.
    :param classifier:      Classifier for predicting best actions without missingness.
                            If none, cleanLoader is assumed best actions.
    :param rand:            Random state.
    :param device:          Device to use for calculations
    :return:  Computed missing value calibration error
    """
    time = perf_counter()
    bucketSizes = torch.zeros((buckets,), dtype=torch.int, device=device)
    # consistency and confidence is per bin
    bucketConfidence = torch.zeros((buckets,), dtype=torch.float, device=device)
    bucketSoftConsistency = torch.zeros((buckets,), dtype=torch.float, device=device)
    bucketHardConsistency = torch.zeros((buckets,), dtype=torch.float, device=device)

    # best actions is per action size
    trueLabelCount = torch.zeros_like(actions, dtype=torch.int)
    bestActionCount = torch.zeros_like(actions, dtype=torch.int)
    softPredictedActionCount = torch.zeros_like(actions, dtype=torch.int)
    hardPredictedActionCount = torch.zeros_like(actions, dtype=torch.int)
    actionsLen = len(actions)
    aleatoricIndex = actionsLen - 1  # TODO: this is messy

    # accuracy is just over whole set
    softAccurateSamples = torch.zeros((1,), dtype=torch.int, device=device)
    hardAccurateSamples = torch.zeros((1,), dtype=torch.int, device=device)

    isLossList = isinstance(lossFunction, List)
    for i, (cleanBatch, mutatedBatch) in enumerate(zip(cleanLoader, mutatedLoader)):
        # Below we are sampling multiple loss functions for post hoc calibration
        batchLoss: callable
        if isLossList:
            randIndex = torch.randint(0, len(lossFunction), (1,), generator=rand)
            batchLoss = lossFunction[randIndex.item()]
        else:
            batchLoss = lossFunction

        # TODO: consider if we want to allow non-categorical true labels, would make more sense if the weights matrix is done
        mutatedFeatures = mutatedBatch[0]
        trueLabels = mutatedBatch[1].to(device=device, dtype=torch.int)
        if len(trueLabels.shape) > 1:
            trueLabels = trueLabels.squeeze(1)
            if len(trueLabels.shape) > 1:
                logging.error(f"Invalid shape for labels, expect single class output: {trueLabels.shape}")
        sampleIndices: Optional[Tensor] = None
        supportedIndices: Tensor
        # if we have indices, ensure they match then pass them along
        if len(mutatedBatch) == 3:
            sampleIndices = mutatedBatch[2]
            # ensure that all samples in this batch can be processed, lets us skip non-cached images when using a cache
            supportedIndices = decisionMaker.supportsIndices(sampleIndices)

            # debug which samples are skipped
            # TODO: can we support integer tensors here?
            if supportedIndices.dtype == torch.bool and torch.count_nonzero(~supportedIndices) != 0:
                logging.warn(f"Skipping samples at indices {sampleIndices[~supportedIndices]}, unsupported by decision maker")

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
            assert cleanFeatures.shape[0] == mutatedFeatures.shape[0], \
                "Clean and mutated dataset must have the same size"
            bestActions = bestActionWithoutMissing(cleanFeatures, classifier, batchLoss, actions)

        # compute predicted actions
        if device is not None:
            mutatedFeatures = mutatedFeatures.to(device)
        hardPred, confidences, softPred = decisionMaker.estimateBestAction(
            mutatedFeatures, batchLoss, actions, rand=rand, indices=sampleIndices, returnBestClass=True
        )

        # map the confidence values to the bucket index
        bucketIndices = (confidences * buckets).int()
        # any confidence of 1.0 gets mapped to max bucket
        bucketIndices[bucketIndices == buckets] = buckets - 1

        # accuracy metric: matches the true label
        softAccuracy = torch.eq(softPred, trueLabels)
        hardAccuracy = torch.eq(hardPred, trueLabels)
        # track total accuracy
        softAccurateSamples += torch.sum(softAccuracy)
        hardAccurateSamples += torch.sum(hardAccuracy)

        # consistency metric: actions match best actions
        softConsistency = torch.eq(softPred, bestActions)
        hardConsistency = torch.eq(hardPred, bestActions)

        # map -1 to max+1 for aleatoric actions so we can count those
        trueLabelCount += trueLabels.bincount(minlength=actionsLen)
        bestActions[bestActions == -1] = aleatoricIndex
        bestActionCount += bestActions.bincount(minlength=actionsLen)
        softPredictedActionCount += softPred.bincount(minlength=actionsLen)
        hardPredictedActionCount += hardPred.bincount(minlength=actionsLen)

        # bucketSizes = indices.bincount(minlength = buckets)
        for bucket in range(buckets):
            bucketMask = torch.eq(bucketIndices, bucket)
            bucketSize = torch.count_nonzero(bucketMask)
            bucketSizes[bucket] += bucketSize
            if bucketSize > 0:
                bucketConfidence[bucket] += confidences[bucketMask].sum()
                bucketSoftConsistency[bucket] += torch.count_nonzero(softConsistency[bucketMask])
                bucketHardConsistency[bucket] += torch.count_nonzero(hardConsistency[bucketMask])

    # calculate full dataset statistics
    totalSamples = bucketSizes.sum()
    averageConfidence = bucketConfidence.sum() / totalSamples
    softAccuracy = softAccurateSamples / totalSamples
    hardAccuracy = hardAccurateSamples / totalSamples
    softConsistency = bucketSoftConsistency.sum() / totalSamples
    hardConsistency = bucketHardConsistency.sum() / totalSamples

    # up until now, bucketConfidence and bucketConsistency have been sums, need to divide by total size for prob
    # need to be careful about divide by zero though, so skip empty buckets
    nonZero = torch.ne(bucketSizes, 0)
    bucketSizes = bucketSizes[nonZero]
    bucketConfidence = bucketConfidence[nonZero] / bucketSizes
    bucketSoftConsistency = bucketSoftConsistency[nonZero] / bucketSizes
    bucketHardConsistency = bucketHardConsistency[nonZero] / bucketSizes

    softMvce = (bucketSizes * torch.abs(bucketSoftConsistency - bucketConfidence)).sum() / totalSamples
    hardMvce = (bucketSizes * torch.abs(bucketHardConsistency - bucketConfidence)).sum() / totalSamples
    time = perf_counter() - time
    logging.info(f"""
        Results for {decisionMaker.name} in {time} seconds with {buckets} buckets:
        * Soft MVCE: {softMvce.cpu().item()}
        * Hard MVCE: {hardMvce.cpu().item()}
        * Average confidence: {averageConfidence.cpu().item()}
        * Average soft accuracy: {softAccuracy.cpu().item()}
        * Average hard accuracy: {hardAccuracy.cpu().item()}
        * Average soft consistency: {softConsistency.cpu().item()}
        * Average hard consistency: {hardConsistency.cpu().item()}
        
        * Non-zero buckets: {torch.nonzero(nonZero).squeeze().cpu()}
        * Final bucket sizes: {bucketSizes.cpu()} totaling {totalSamples.cpu().item()} samples
        * Final bucket confidences: {bucketConfidence.cpu()}
        * Final bucket soft consistencies: {bucketSoftConsistency.cpu()}
        * Final bucket hard consistencies: {bucketHardConsistency.cpu()}
        
        * Actions: {actions.cpu()}
        * True Label Counts: {trueLabelCount.cpu()}
        * Best Action Counts: {bestActionCount.cpu()}
        * Prediction Soft Action Counts: {softPredictedActionCount.cpu()}
        * Prediction Hard Action Counts: {hardPredictedActionCount.cpu()}
    """)

    # return all results, they can use named tuple syntax to select relevant results
    return MVCEResults(
        mvce=SoftHardTensor(softMvce, hardMvce),
        confidence=averageConfidence,
        accuracy=SoftHardTensor(softAccuracy, hardAccuracy),
        consistency=SoftHardTensor(softConsistency, hardConsistency)
    )

def computeECE(loader: DataLoader, classifier: Regressor, classCount: int, buckets: int,
               device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """
    Computes the expected calibration error for the given classifier.
    :param loader:          DataLoader for data with no missingness.
    :param classifier:      Classifier to test.
    :param buckets:         Number of buckets for computing the calibration error.
    :param classCount:      Number of expected features
    :param device:          Device to use for calculations
    :return:  Computed expected calibration error and accuracy
    """
    time = perf_counter()
    bucketSizes = torch.zeros((buckets,), dtype=torch.int, device=device)
    bucketConfidence = torch.zeros((buckets,), dtype=torch.float, device=device)
    bucketAccuracy = torch.zeros((buckets,), dtype=torch.float, device=device)

    actionCount = classCount if classCount > 1 else 2
    labelCount = torch.zeros((actionCount,), dtype=torch.int, device=device)
    predictedCount = torch.zeros((actionCount,), dtype=torch.int, device=device)

    accurateSamples = torch.zeros((1,), dtype=torch.int, device=device)

    for i, (features, labels) in enumerate(loader):
        features: Tensor
        labels: Tensor = labels.squeeze()
        if device is not None:
            features = features.to(device)
            labels = labels.to(device)
        phi = classifier.predict(features).squeeze()

        # if we only have 1 class, threshold to 0.5
        prediction: Tensor
        confidence: Tensor
        if len(phi.shape) == 1 or phi.shape[1] == 1:
            prediction = (phi >= 0.5).int()
            confidence = (phi * prediction + (1 - phi) * (1 - prediction))
        else:
            max = phi.max(1)
            confidence = max.values
            prediction = max.indices

        # map the confidence values to the bucket index
        bucketIndices = (confidence * buckets).int()
        # any confidence of 1.0 gets mapped to max bucket
        bucketIndices[bucketIndices == buckets] = buckets - 1

        # accuracy metric: predicted label matches actual
        accuracy = torch.eq(prediction, labels)
        # track total accuracy
        accurateSamples += torch.sum(accuracy)

        labelCount += labels.int().bincount(minlength=actionCount)
        predictedCount += prediction.bincount(minlength=actionCount)

        # bucketSizes = indices.bincount(minlength = buckets)
        for bucket in range(buckets):
            bucketMask = bucketIndices == bucket
            bucketSize = torch.count_nonzero(bucketMask)
            bucketSizes[bucket] += bucketSize
            if bucketSize > 0:
                bucketConfidence[bucket] += confidence[bucketMask].sum()
                bucketAccuracy[bucket] += torch.count_nonzero(accuracy[bucketMask])

    # up until now, bucketConfidence and bucketConsistency have been sums, need to divide by total size for prob
    # need to be careful about divide by zero though, so skip empty buckets
    nonZero = torch.ne(bucketSizes, 0)
    bucketSizes = bucketSizes[nonZero]
    bucketConfidence = bucketConfidence[nonZero] / bucketSizes
    bucketAccuracy = bucketAccuracy[nonZero] / bucketSizes

    totalSamples = bucketSizes.sum()
    ece = (bucketSizes * torch.abs(bucketAccuracy - bucketConfidence)).sum() / totalSamples
    accuracy = accurateSamples / totalSamples
    time = perf_counter() - time
    logging.info(f"""
        Computed ECE {ece.cpu().item()} in {time} seconds with {buckets} buckets:
        * Non-zero buckets: {torch.nonzero(nonZero).squeeze().cpu()}
        * Final bucket sizes: {bucketSizes.cpu()} totaling {totalSamples.cpu().item()} samples
        * Final bucket confidences: {bucketConfidence.cpu()}
        * Final bucket accuracy: {bucketAccuracy.cpu()}
        * Label counts: {labelCount.cpu()}
        * Prediction Action Counts: {predictedCount.cpu()}
        * Average accuracy: {accuracy.cpu().item()}
    """)

    # return final values
    return ece, accuracy


class MVCEExperiment:
    # mvce parameters, see `computeMVCE` for docs
    cleanLoader: DataLoader
    """Data loader for clean samples"""
    mutatedLoader: DataLoader
    """Data loader for mutated samples"""
    decisionMaker: DecisionMaker
    """Logic for making a decision"""
    lossFunction: callable
    """User provided cost function, simplest case is zero-one"""
    actions: Tensor
    """User provided action tensor, may match up to classes"""
    buckets: int
    """Number of buckets for MVCE to use"""
    classifier: Optional[Regressor]
    """Classifier for clean predictions, may be None if `cleanLoader` provides clean actions."""
    rand: Generator
    """Random state for consistency in experiments"""
    device: torch.device
    """Device to run experiments"""

    # additional parameters
    maskName: str
    """Name of the missing region"""
    actionName: str
    """Name of the action space"""
    trials: int
    """Number of times to compute the MVCE, for the sake of error bars"""
    time: Optional[float]
    """Duration of this experiment"""
    mvce: SoftHardTensor
    """MVCE results for this experiment, size is equal to trials"""
    accuracies: SoftHardTensor
    """Accuracy percentage for the dataset, size is equal to trials"""
    consistencies: SoftHardTensor
    """Consistency percentage for the dataset, size is equal to trials"""
    confidences: Tensor
    """Average confidence for the dataset, size is equal to trials"""

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
            # initialize results space
            def emptyTrials() -> Tensor:
                return torch.empty((self.trials,), dtype=torch.float)
            self.mvce = SoftHardTensor(emptyTrials(), emptyTrials())
            self.accuracies = SoftHardTensor(emptyTrials(), emptyTrials())
            self.consistencies = SoftHardTensor(emptyTrials(), emptyTrials())
            self.confidences = emptyTrials()
            for i in range(self.trials):
                 results = computeMVCE(
                    self.cleanLoader, self.mutatedLoader, self.decisionMaker,
                    self.lossFunction, self.actions, self.buckets, self.classifier,
                    self.rand, self.device)
                 # store each result into the per trial vectors, for averaging later
                 self.mvce.soft[i] = results.mvce.soft.cpu()
                 self.mvce.hard[i] = results.mvce.hard.cpu()
                 self.confidences[i] = results.confidence.cpu()
                 self.accuracies.soft[i] = results.accuracy.soft.cpu()
                 self.accuracies.hard[i] = results.accuracy.hard.cpu()
                 self.consistencies.soft[i] = results.consistency.soft.cpu()
                 self.consistencies.hard[i] = results.consistency.hard.cpu()
        except KeyboardInterrupt as e:
            # this is just logging the context so we know which experiment was terminated
            # its in the log again later and earlier, but this reduces some of the debug time
            logging.error(f"Received keyboard interrupt during {self.experimentName}, terminating program")
            raise e
        except BaseException as e:
            handleException(type(e), e, e.__traceback__,
                            message=f"Failed to process {self.experimentName}")
            return

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
            "Method", "Action Space", "Mask", "Time", "Scale",
            # MVCE
            "Soft MVCE Mean", "Soft MVCE Std",
            "Hard MVCE Mean", "Hard MVCE Std",
            # Accuracy
            "Soft Accuracy Mean", "Soft Accuracy Std",
            "Hard Accuracy Mean", "Hard Accuracy Std",
            # Consistency & Confidence
            "Soft Consistency Mean", "Soft Consistency Std",
            "Hard Consistency Mean", "Hard Consistency Std",
            "Confidence Mean", "Confidence Std",

            # Trials - MVCE
            *[f"Trial {i+1} Soft MVCE" for i in range(trials)],
            *[f"Trial {i+1} Hard MVCE" for i in range(trials)],
            # Trials - Accuracy
            *[f"Trial {i+1} Soft Accuracy" for i in range(trials)],
            *[f"Trial {i+1} Hard Accuracy" for i in range(trials)],
            # Trials - Consistency & Confidence
            *[f"Trial {i+1} Soft Consistency" for i in range(trials)],
            *[f"Trial {i+1} Hard Consistency" for i in range(trials)],
            *[f"Trial {i+1} Confidence" for i in range(trials)]
        ])

    def writeResults(self, csvFile) -> None:
        """
        Writes the results to the file
        :param csvFile:  CSV file for result writing
        """
        if self.time is None:
            logging.error(f"Skipping including {self.experimentName} in result CSV as it did not complete.")

        csvFile.writerow([
            self.decisionMaker.name, self.actionName, self.maskName, self.time, self.decisionMaker.scale,
            # MVCE
            self.mvce.soft.mean().item(), self.mvce.soft.std().item(),
            self.mvce.hard.mean().item(), self.mvce.hard.std().item(),
            # Accuracy
            self.accuracies.soft.mean().item(), self.accuracies.soft.std().item(),
            self.accuracies.hard.mean().item(), self.accuracies.hard.std().item(),
            # Consistency & Confidence
            self.consistencies.soft.mean().item(), self.consistencies.soft.std().item(),
            self.consistencies.hard.mean().item(), self.consistencies.hard.std().item(),
            self.confidences.mean().item(), self.confidences.std().item(),

            # Trials - MVCE
            *[result.item() for result in self.mvce.soft],
            *[result.item() for result in self.mvce.hard],
            # Trials - Accuracy
            *[result.item() for result in self.accuracies.soft],
            *[result.item() for result in self.accuracies.hard],
            # Trials - Consistency & Confidence
            *[result.item() for result in self.consistencies.soft],
            *[result.item() for result in self.consistencies.hard],
            *[result.item() for result in self.confidences]
        ])

# post-hoc
class CalibrationScaleExperiment:
    # mvce parameters, see `computeMVCE` for docs
    cleanLoader: DataLoader
    """Data loader for clean samples"""
    mutatedLoader: DataLoader
    """Data loader for mutated samples"""
    decisionMaker: DecisionMaker
    """Logic for making a decision"""
    lossFunctions: List[callable]
    """List of loss functions to be used for taking Expectation when calculating post hoc callibration"""
    actions: Tensor
    """User provided action tensor, may match up to classes"""
    buckets: int
    """Number of buckets for MVCE to use"""
    classifier: Optional[Regressor]
    """Classifier for clean predictions, may be None if `cleanLoader` provides clean actions."""
    rand: Generator
    """Random state for consistency in experiments"""
    device: torch.device
    """Device to run experiments"""

    # additional parameters
    maskName: str
    """Name of the missing region"""
    trials: int
    """Number of times to compute the MVCE, for the sake of error bars"""
    time: Optional[float]
    """Duration of this experiment"""
    mvce: SoftHardTensor
    """MVCE results for this experiment, size is equal to trials"""

    def __init__(self, cleanLoader: DataLoader, maskName: str, mutatedLoader: DataLoader, decisionMaker: DecisionMaker,
                 lossFunctions: List[callable], actions: Tensor, buckets: int, trials: int,
                 classifier: Regressor = None, rand: Generator = None, device: Optional[torch.device] = None):
        self.cleanLoader = cleanLoader
        self.maskName = maskName
        self.mutatedLoader = mutatedLoader
        self.decisionMaker = decisionMaker

        self.lossFunctions = lossFunctions
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
        return f"{self.decisionMaker.name} missing {self.maskName} at scale {self.decisionMaker.scale}"

    def __call__(self, *args, **kwargs):
        logging.info(f"Started running {self.experimentName}")
        startTime = perf_counter()

        try:
            self.mvce = SoftHardTensor(
                torch.empty((self.trials,), dtype=torch.float),
                torch.empty((self.trials,), dtype=torch.float)
            )
            for i in range(self.trials):
                result = computeMVCE(
                    self.cleanLoader, self.mutatedLoader, self.decisionMaker,
                    self.lossFunctions, self.actions, self.buckets, self.classifier,
                    self.rand, self.device
                )
                self.mvce.soft[i] = result.mvce.soft.cpu()
                self.mvce.hard[i] = result.mvce.hard.cpu()
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
            "Method", "Mask", "Time", "Scale",
            "Soft MVCE Mean", "Soft MVCE Std",
            "Hard MVCE Mean", "Hard MVCE Std",
            *[f"Trial {i+1} Soft MVCE" for i in range(trials)],
            *[f"Trial {i+1} Hard MVCE" for i in range(trials)]
        ])

    def writeResults(self, csvFile) -> None:
        """
        Writes the results to the file
        :param csvFile:  CSV file for result writing
        """
        if self.time is None:
            logging.error(f"Skipping including {self.experimentName} in result CSV as it did not complete.")

        csvFile.writerow([
            self.decisionMaker.name, self.maskName, self.time, self.decisionMaker.scale,
            self.mvce.soft.mean().item(), self.mvce.soft.std().item(),
            self.mvce.hard.mean().item(), self.mvce.hard.std().item(),
            *[result.item() for result in self.mvce.soft],
            *[result.item() for result in self.mvce.hard]
        ])