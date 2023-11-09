import argparse
import csv
import logging
import os
from threading import Thread
from time import perf_counter
from typing import List

from torch import Generator

from mvu.dataset import DatasetSplits
from mvu.distribution import ConditionalGaussianDistribution, Distribution, MarginalGaussianDistribution
from mvu.experiment import Experiment
from mvu.imputator import ZeroImputator, ConstantImputator, Imputator
from mvu.logger import setupLogging
from mvu.method import Method, BasicCombinationMethod, EmpiricalUncertaintyByCount, EmpiricalUncertaintyByFeature, \
    MonteCarloMethod
from mvu.regressor import Regressor
from mvu.util import estimateResidual
from mvu.threading import WorkQueue, Worker, distributeTasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=str, default=None, help='Path to the pretrained regressor to load')
    parser.add_argument("--regressor", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--missing", type=float, nargs='*', help="Percent of data to treat as missing")
    parser.add_argument("--mc_samples", type=int, nargs='*', default=[],
                        help="Number of Monte Carlo samples to take")

    parser.add_argument('--seed', type=int, default=1337,
                        help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)

    # load in regressor
    logging.info(f"Loading regressor from {args.regressor}")
    regressor = Regressor.load(args.regressor)

    # load in dataset
    datasetPath = args.dataset
    if datasetPath is None:
        datasetPath = f"./datasets/binary/{args.name}.pklz"
    logging.info(f"Loading dataset from {args.regressor}")
    ds = DatasetSplits.load(datasetPath)

    # compute residual, it is just a function of regressor and dataset so only need one
    startTime = perf_counter()
    residual = estimateResidual(regressor, ds.validate)
    endTime = perf_counter()
    logging.info(f"Computed residual uncertainty of {residual}. Took {endTime - startTime}")

    # setup experiments
    methods: List[Method] = []
    rand = Generator()
    rand.manual_seed(args.seed)

    def method(method: Method):
        """Adds an method"""
        methods.append(method)

    def imputator(imputator: Imputator):
        """Adds all three basic imputation methods"""
        method(BasicCombinationMethod(regressor, imputator))
        method(EmpiricalUncertaintyByCount(regressor, imputator, ds.validate, residual, rand))
        method(EmpiricalUncertaintyByFeature(regressor, imputator, ds.validate, residual))

    def monteCarlo(distribution: Distribution):
        for samples in args.mc_samples:
            method(MonteCarloMethod(regressor, distribution, samples, rand))

    gaussian = ConditionalGaussianDistribution.fromDataset(ds.validate)
    # basic
    imputator(ZeroImputator())
    imputator(ConstantImputator(ds.metadata.normalizeFeatures(gaussian.mean), "Mean"))
    imputator(gaussian)  # Gaussian Conditional Mean Imputation
    # monte carlo
    monteCarlo(MarginalGaussianDistribution.fromGaussian(gaussian))
    monteCarlo(gaussian)

    # setup experiments list
    experiments: List[Experiment] = []
    totalFeatures = ds.metadata.numGroups
    for missing in args.missing:
        missingTest = ds.test.dropCount(int(totalFeatures*missing), rand=rand)
        for method in methods:
            experiments.append(Experiment(method, missingTest, missing, residual))

    # if -1, give each experiment its own thread
    distributeTasks(experiments, args.threads)

    # save all experiment results to the relevant CSV files
    outputName = f"{args.name}-{date}"
    with open(os.path.join(outputFolder, f"{outputName}-summary.csv"), "w") as summaryFile:
        with open(os.path.join(outputFolder, f"{outputName}-all.csv"), "w") as allFile:
            # summary CSV has one row per experiment
            summaryCsv = csv.writer(summaryFile)
            # all CSV has one row per sample
            allCsv = csv.writer(allFile)

            # write headers
            Experiment.writeResultHeaders(summaryCsv, allCsv)
            # write rows
            for experiment in experiments:
                experiment.writeResults(summaryCsv, allCsv)
