import logging
from time import perf_counter
from typing import Optional, List, Tuple

import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .dataset.csv import CsvDataset
from .logger import handleException
from .model.method import Method
from .util import gaussianLogLikelihood


class Experiment:
    """
    Experiment that runs the entire dataset as a single batch.
    Notably supports outputting results for each sample separately.
    Does not scale to sets with more samples but tends to be faster. Most likely you want batched experiment.
    """

    # Basic inputs
    method: Method
    """Method for primary part of the experiment"""
    residual: Tensor
    """Previously computed residual uncertainty, tensor of size 1"""
    missingName: str
    """Name of the missing experiment, written to CSV if missingPercent is None"""
    missingPercent: Optional[float]
    """Missing percent between 0.0 and 1.0"""
    rand: Optional[Generator]
    """Allows us to guarantee no matter what order experiments run, we still get the same results when seeded"""
    storeAllResults: bool
    """If true, stores per sample results as a tensor in the experiment."""
    device: Optional[torch.device]
    """Device used for experiments, if None uses default device (typically CPU)"""

    # data loading
    dataName: str
    """Name of the dataset to use"""
    data: Optional[DataLoader]
    """
    Loader to fetch the data for the experiments. Either this or dataset must be set.
    It is expected when enumerating to get a tensor of `(features,targets)`
    """
    dataset: Optional[CsvDataset]
    """Dataset for the current experiment. Either this or data must be set."""

    # Ongoing results
    totalSamples: int
    """Number of samples we have seen from the dataset"""
    processedSamples: int
    """Number of samples that succeeded from the dataset, should be less than or equal to `totalSamples`"""
    squaredError: Tensor
    """Sum of the squared error. Divide by `processedSamples` to get MSE."""
    missingVariance: Optional[Tensor]
    """Sum of missing variance so far. Divide by `processedSamples` to get average variance."""
    ll: Tensor
    """Sum of log likelihood so far, divide by `processedSamples` to get average LL."""
    sampleResults: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
    """
    List of experiment results for each batch, to save to the final CSV file.
    Tensors in order are target, mean, variance, squared error, and log likelihood.
    """
    """List of experiment missing variances for each batch, will be concatenated for final output."""
    time: float
    """Duration of this experiment"""

    def __init__(self, method: Method, dataName: str, missingName: str, missingPercent: float = None,
                 residual: Tensor = None, data: DataLoader = None, dataset: CsvDataset = None,
                 storeAllResults: bool = False, rand: Generator = None, device: torch.device = None):
        assert data is not None or dataset is not None, "Must pass in either data or dataset"
        self.method = method
        self.dataName = dataName
        self.missingName = missingName
        self.missingPercent = missingPercent
        self.residual = residual if residual is not None else torch.tensor([0], device=device, dtype=torch.float)
        self.storeAllResults = storeAllResults
        self.device = device

        self.data = data
        self.dataset = dataset
        self.rand = rand
        # results
        self.totalSamples = 0
        self.processedSamples = 0
        self.squaredError = torch.tensor([0], device=device, dtype=torch.float)
        self.missingVariance = torch.tensor([0], device=device, dtype=torch.float)
        self.ll = torch.tensor([0], device=device, dtype=torch.float)
        self.sampleResults = []
        self.time = 0

    @property
    def experimentName(self):
        """Name of the overall experiment"""
        return f"{self.dataName} - {self.method.name} - {self.missingName}"

    def _runBatch(self, features: Tensor, targetsCpu: Tensor, details: str = "") -> float:
        """
        Runs a single batch of the experiment
        :param features:    Features for this batch
        :param targetsCpu:  Targets for this batch, to compute squared error and log likelihood
        :param details:     Extra information for exception debugging
        :return:   Time this batch took to run
        """
        batchStart = perf_counter()
        self.totalSamples += targetsCpu.shape[0]
        try:
            if self.device is not None:
                features = features.to(self.device)
                targets = targetsCpu.to(self.device)

            mean, variance = self.method.predictWithUncertainty(features, self.rand)

            # we use 2 summary statistics: squared error and LL, save them both ready for averaging
            squaredError = (mean - targets) ** 2
            totalVariance = variance + self.residual
            ll = gaussianLogLikelihood(squaredError, totalVariance)

            # store the results for this batch if requested
            # TODO: consider supporting separate output CSV per experiment so we don't have to store this all in memory
            if self.storeAllResults:
                self.sampleResults.append((targetsCpu, mean.cpu(), variance.cpu(), squaredError.cpu(), ll.cpu()))

            # start summing results
            self.squaredError += squaredError.sum()
            self.missingVariance += variance.sum()
            self.ll += ll.sum()
            self.processedSamples += targetsCpu.shape[0]
            batchEnd = perf_counter()
            return batchEnd - batchStart
        except KeyboardInterrupt as e:
            raise e  # propagate keyboard interrupt
        except BaseException as e:
            batchEnd = perf_counter()
            time = batchEnd - batchStart
            handleException(type(e), e, e.__traceback__,
                            message=f"Failed to process {self.experimentName}{details} in {time} seconds")
            return time

    def __call__(self, *args, **kwargs):
        """Runs the main experiment, will happen during threading"""
        logging.info(f"Started running {self.experimentName}")
        startTime = perf_counter()

        # approach 1: data loader, run experiment in batches
        try:
            if self.data is not None:
                try:
                    for i, (features, targets) in enumerate(self.data):
                        time = self._runBatch(features, targets, f" batch {i+1}")
                        logging.debug(f"Batch {i+1} for {self.experimentName} done in {time} seconds")
                except KeyboardInterrupt as e:
                    raise e  # propagate keyboard interrupt
                except BaseException as e:
                    handleException(type(e), e, e.__traceback__,
                                    message=f"Failed to process {self.experimentName} due to dataloader exception")
            elif self.dataset is not None:
                self._runBatch(self.dataset.features, self.dataset.targets)
        except KeyboardInterrupt as e:
            # this is just logging the context so we know which experiment was terminated
            # its in the log again later and earlier, but this reduces some of the debug time
            logging.error(f"Received keyboard interrupt during {self.experimentName}, terminating program")
            raise e

        # store final experiment time
        endTime = perf_counter()
        self.time = endTime - startTime
        logging.info(f"Finished running {self.experimentName} in {self.time} seconds")

    @classmethod
    def writeResultHeaders(cls, summaryCsv, allCsv = None) -> None:
        """
        Writes the headers for both result CSV files
        :param summaryCsv:  CSV file for experiment summaries
        :param allCsv:      CSV file for per sample results, if None skipped
        """
        summaryCsv.writerow([
            "Name", "Missing", "Runtime",
            "Missing Variance", "Residual", "Total Variance",
            "MSE", "LL",
            "Processed Samples", "Total Samples"
        ])
        if allCsv is not None:
            allCsv.writerow([
                "Name", "Missing", "Sample",
                "Expected", "Mean",
                "Missing Variance", "Residual", "Total Variance",
                "Squared Error", "LL"
            ])

    def writeResults(self, summaryCsv, allCsv = None):
        """
        Writes the results for this experiment to the CSV files
        :param summaryCsv:  CSV file for experiment summaries
        :param allCsv:      CSV file for per sample results, if None skipped
        """
        # write missingPercent if not None, else write missing
        missing = self.missingPercent if self.missingPercent is not None else self.missingName
        if self.processedSamples == 0:
            logging.error(f"Skipping including {self.experimentName} in result CSV as 0/{self.totalSamples} samples "
                          "were processed.")

        # start by writing the summary row
        avgMissingVariance = self.missingVariance / self.processedSamples
        summaryCsv.writerow([
            self.method.name, missing, self.time,
            avgMissingVariance.item(), self.residual.item(), (avgMissingVariance + self.residual).item(),
            (self.squaredError / self.processedSamples).item(), (self.ll / self.processedSamples).item(),
            self.processedSamples, self.totalSamples
        ])

        # if requested, store all results
        if self.storeAllResults and allCsv is not None:
            # then write a row for each sample
            offset = 0
            for batch in self.sampleResults:
                for batchIndex, (target, mean, missingVariance, squaredError, ll) in enumerate(zip(*batch)):
                    allCsv.writerow([
                        self.method.name, missing, batchIndex + offset,
                        target.item(), mean.item(),
                        missingVariance.item(), self.residual.item(), (missingVariance + self.residual).item(),
                        squaredError.item(), ll.item()
                    ])
                offset += len(batch[0])
        elif self.storeAllResults:
            logging.error(f"Experiment {self.experimentName} stored all results, but did not receive CSV to write them.")
        elif allCsv is not None:
            logging.error(f"Experiment {self.experimentName} received CSV file for all results but did not store them.")


def appendExperiments(experiments: List[Experiment], methods: List[Method],
                      *args, rand: Generator = None, **kwargs) -> None:
    """
    Appends an experiment for each method in a set
    :param experiments:     List of experiments, will be modified
    :param methods:         List of methods to pull from
    :param rand:            Rand seed
    """
    # give each experiment its own random state,
    # goal is to ensure reproducibility despite the fact the order tasks run is non-deterministic
    # TODO: should these just all use the same seed but their own rand copy?
    seeds = torch.randint(0, 0x7fffffff, (len(methods),), generator=rand)  # max is just 32-bit signed int max
    for (method, seed) in zip(methods, seeds):
        newRand = torch.Generator()
        newRand.manual_seed(seed.item())
        experiments.append(Experiment(method, *args, rand=newRand, **kwargs))


def appendDatasetAsBatchExperiments(experiments: List[Experiment], methods: List[Method], dataset: CsvDataset,
                                    *args, batchSize: int = 1, **kwargs) -> None:
    """
    Appends an experiment for each method in a set. Additional arguments are passed on.
    :param experiments:     List of experiments, will be modified
    :param methods:         List of methods to pull from
    :param dataset:         Dataset for each experiment
    :param batchSize:       Number of samples per batch
    """
    dataLoader = DataLoader(dataset.toTorch(), batch_size=batchSize, shuffle=False)
    appendExperiments(experiments, methods, dataset.metadata.name, *args, data=dataLoader, **kwargs)
