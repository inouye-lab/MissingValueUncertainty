import argparse
import csv
import logging
import os
from time import perf_counter
from typing import List

import torch
from torch import Generator

from mvu.dataset import DatasetSplits
from mvu.distribution import ConditionalGaussianDistribution, Distribution, MarginalGaussianDistribution
from mvu.experiment import Experiment, appendExperiments
from mvu.imputator import ZeroImputator, ConstantImputator, Imputator, MiceImputator
from mvu.logger import setupLogging
from mvu.method import Method, BasicCombinationMethod, EmpiricalUncertaintyByCount, EmpiricalUncertaintyByFeature, \
    MonteCarloMethod
from mvu.regressor import Regressor
from mvu.util import estimateResidual
from mvu.threading import distributeTasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=str, default=None, help='Path to processed dataset to load')
    parser.add_argument("--regressor", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--mc_samples", type=int, nargs='*', default=[],
                        help="Number of Monte Carlo samples to take")
    parser.add_argument("--mice_iterations", type=int, nargs='*', default=[],
                        help="Number of mice iterations to run")

    parser.add_argument("--missing", type=float, default=[], nargs='*',
                        help="Percent of data to treat as missing. If undefined, runs no missing percent experiments")
    parser.add_argument("--feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature separately missing.')
    parser.add_argument("--inverted_feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature only present.')

    parser.add_argument("--gaussian_pseudo_inverse", action='store_true',
                        help='If set, uses the pseudo-inverse for multiplications for the gaussian methods.'
                             'If unset, uses the least squares approach.')
    parser.add_argument("--gaussian_schur", action='store_true',
                        help='If set, uses the schur complement to compute the gaussian covariance matrix.'
                             'If unset, uses matrix multiplications respecting gaussian_pseudo_inverse')

    parser.add_argument('--seed', type=int, default=1337,
                        help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)

    # validate arguments
    assert len(args.missing) > 0 or args.feature_impact or args.inverted_feature_impact, \
        "Must either run feature impact or missing percent"

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
        method(EmpiricalUncertaintyByCount(regressor, imputator, ds.validate, residual))
        method(EmpiricalUncertaintyByFeature(regressor, imputator, ds.validate, residual))

    def monteCarlo(distribution: Distribution):
        for samples in args.mc_samples:
            method(MonteCarloMethod(regressor, distribution, samples))

    gaussian = ConditionalGaussianDistribution.fromDataset(
        ds.train, schur=args.gaussian_schur, leastSquares=not args.gaussian_pseudo_inverse
    )
    # basic
    imputator(ZeroImputator())
    imputator(ConstantImputator(ds.metadata.normalizeFeatures(gaussian.mean), "Mean"))
    imputator(gaussian)  # Gaussian Conditional Mean Imputation
    # monte carlo
    monteCarlo(MarginalGaussianDistribution.fromGaussian(gaussian))
    monteCarlo(gaussian)
    # mice
    for iterations in args.mice_iterations:
        # some of the non-augmented MICE will fail, but the experiments are setup to handle that
        imputator(MiceImputator(ds.metadata, iterations))
        imputator(MiceImputator(ds.metadata, iterations, ds.train.features, "Training Features"))

    # setup experiments list
    totalFeatures = ds.metadata.numGroups
    expName = ""
    if len(args.missing) > 0:
        expName += f", missing percentages {args.missing}"
    if args.feature_impact:
        expName += f", feature impact over {totalFeatures} features"
    if args.inverted_feature_impact:
        expName += f", inverted feature impact over {totalFeatures} features"
    logging.info(f"Setting up experiments with {len(methods)} methods{expName}")
    experiments: List[Experiment] = []

    # missing percentage experiments
    for missing in args.missing:
        # all experiments use the same missing values
        missingName = f"{int(missing*100)}% missing"
        logging.info(f"Setting up experiments for {missingName}")
        missingTest = ds.test.dropCount(int(totalFeatures*missing), rand=rand)
        appendExperiments(experiments, methods, missingTest, missingName, missing, residual, rand)

    # individual missing feature experiment
    if args.feature_impact or args.inverted_feature_impact:
        # get group indices for dropping
        groups = ds.metadata.groups
        if groups is None:
            groups = torch.arange(0, totalFeatures)

        if args.feature_impact:
            # create experiment for each feature
            for index in range(totalFeatures):
                featureName = ds.metadata.featureName(index)
                logging.info(f"Setting up experiments for '{featureName}'")

                missingTest = ds.test.dropSpecified(torch.eq(groups, index))
                appendExperiments(experiments, methods, missingTest, featureName, None, residual, rand)

        if args.inverted_feature_impact:
            # create experiment for each feature
            for index in range(totalFeatures):
                featureName = "not " + ds.metadata.featureName(index)
                logging.info(f"Setting up experiments for '{featureName}'")

                missingTest = ds.test.dropSpecified(torch.ne(groups, index))
                appendExperiments(experiments, methods, missingTest, featureName, None, residual, rand)

    # if -1, give each experiment its own thread
    distributeTasks(experiments, args.threads)
    successful = len([exp for exp in experiments if exp.completed])
    logging.info(f"Finished running {successful}/{len(experiments)} experiments")

    # save all experiment results to the relevant CSV files
    outputName = f"{args.name}-{date}"
    summaryPath = os.path.join(outputFolder, f"{outputName}-summary.csv")
    allPath = os.path.join(outputFolder, f"{outputName}-all.csv")
    logging.info(f"Saving results to {summaryPath} and {allPath}")
    with open(summaryPath, "w") as summaryFile:
        with open(allPath, "w") as allFile:
            # summary CSV has one row per experiment
            summaryCsv = csv.writer(summaryFile)
            # all CSV has one row per sample
            allCsv = csv.writer(allFile)

            # write headers
            Experiment.writeResultHeaders(summaryCsv, allCsv)
            # write rows
            for experiment in experiments:
                experiment.writeResults(summaryCsv, allCsv)
    logging.info("Finished saving results")
