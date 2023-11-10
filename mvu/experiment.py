import logging
from time import perf_counter
from typing import Optional

import torch
from torch import Tensor, Generator

from .dataset import Dataset
from .logger import handleException
from .method import Method
from .util import gaussianLogLikelihood


class Experiment:
    # Inputs
    method: Method
    """Method for primary part of the experiment"""
    dataset: Dataset
    """Dataset for the current experiment"""
    residual: Tensor
    """Previously computed residual uncertainty, tensor of size 1"""
    missingPercent: float
    """Missing percent between 0.0 and 1.0"""
    rand: Optional[Generator]
    """Allows us to guarantee no matter what order experiments run, we still get the same results when seeded"""

    # Results
    completed: bool
    """If true, this thread completed running"""
    mean: Optional[Tensor]
    """Experiment prediction, size is `(samples,)` based on `dataset`"""
    variance: Optional[Tensor]
    """Experiment variance, size is `(samples,)` based on `dataset`"""
    time: float
    """Duration of this experiment"""

    def __init__(self, method: Method, dataset: Dataset, missingPercent: float, residual: Tensor = Tensor([0]),
                 rand: Generator = None):
        self.method = method
        self.dataset = dataset
        self.missingPercent = missingPercent
        self.residual = residual
        self.rand = rand
        # results
        self.completed = False
        self.mean = None
        self.variance = None
        self.time = 0

    @property
    def experimentName(self):
        """Name of the overall experiment"""
        return f"{self.method.name} - {int(self.missingPercent*100)}% missing"

    def __call__(self, *args, **kwargs):
        """Runs the main experiment, will happen during threading"""
        logging.info(f"Started running {self.experimentName}")
        startTime = perf_counter()
        try:
            self.mean, self.variance = self.method.predictWithUncertainty(self.dataset.features, self.rand)
            endTime = perf_counter()
            self.time = endTime - startTime
            self.completed = True
            logging.info(f"Finished running {self.experimentName} in {self.time}")
        except BaseException as e:
            endTime = perf_counter()
            self.time = endTime - startTime
            handleException(type(e), e, e.__traceback__,
                            message=f"Failed to finish {self.experimentName} after {self.time}")

    @classmethod
    def writeResultHeaders(cls, summaryCsv, allCsv):
        """Writes the headers for the CSV result files"""
        summaryCsv.writerow([
            "Name", "Missing Percent", "Runtime",
            "Missing Variance", "Residual", "Total Variance",
            "MSE", "LL"
        ])
        allCsv.writerow([
            "Name", "Missing Percent", "Sample",
            "Expected", "Mean",
            "Missing Variance", "Residual", "Total Variance",
            "Squared Error", "LL"
        ])

    def writeResults(self, summaryCsv, allCsv):
        if not self.completed:
            logging.info(f"Skipping saving {self.experimentName} as it did not complete")
            return

        """Writes the results to the relevant CSV files"""
        # data we have ready: name, missing percent, runtime, missing variance, residual
        # compute remaining data
        squaredError = (self.mean - self.dataset.targets) ** 2
        totalVariance = self.variance + self.residual
        ll = gaussianLogLikelihood(squaredError, totalVariance)

        # start by writing the summary row
        summaryCsv.writerow([
            self.method.name, self.missingPercent, self.time,
            torch.mean(self.variance).item(), self.residual.item(), torch.mean(totalVariance).item(),
            torch.mean(squaredError).item(), torch.mean(ll).item()
        ])
        # then write a row for each sample
        for sampleIndex, (expected, mean, missingVariance, totalVariance, squaredError, ll) \
                in enumerate(zip(self.dataset.targets, self.mean, self.variance, totalVariance, squaredError, ll)):
            allCsv.writerow([
                self.method.name, self.missingPercent, sampleIndex,
                expected.item(), mean.item(),
                missingVariance.item(), self.residual.item(), totalVariance.item(),
                squaredError.item(), ll.item()
            ])
