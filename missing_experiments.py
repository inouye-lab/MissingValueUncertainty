import argparse
import csv
import logging
import os
from time import perf_counter
from typing import List, Optional, TextIO

import torch
from torch import Generator, Tensor
from torch.utils.data import DataLoader

from mvu.dataset.csv import CsvDatasetSplits
from mvu.model.distribution import ConditionalGaussianDistribution, Distribution, MarginalGaussianDistribution
from mvu.experiment import Experiment, appendExperiments
from mvu.model.imputator import ZeroImputator, ConstantImputator, Imputator, MiceImputator
from mvu.logger import setupLogging
from mvu.model.method import Method, BasicCombinationMethod, EmpiricalUncertaintyByCount, EmpiricalUncertaintyByFeature, \
    MonteCarloMethod
from mvu.dataset.mutators import SpecificFeatureRemovingDataset, FeatureCountRemovingDataset
from mvu.model.regressor import Regressor
from mvu.util import estimateResidual
from mvu.threading import distributeTasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=str, default=None, help='Path to processed dataset to load')
    parser.add_argument("--regressor", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')
    parser.add_argument("--write_all_results", action='store_true',
                        help="If set, writes a CSV with results from all samples.")

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--mc_samples", type=int, nargs='*', default=[],
                        help="Number of Monte Carlo samples to take")
    parser.add_argument("--mice_iterations", type=int, nargs='*', default=[],
                        help="Number of mice iterations to run")


    # experiment selection
    parser.add_argument("--missing", type=float, default=[], nargs='*',
                        help="Percent of data to treat as missing. If undefined, runs no missing percent experiments")
    parser.add_argument("--feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature separately missing.')
    parser.add_argument("--inverted_feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature only present.')

    # method configuration
    parser.add_argument("--gaussian_pseudo_inverse", action='store_true',
                        help='If set, uses the pseudo-inverse for multiplications for the gaussian methods.'
                             'If unset, uses the least squares approach.')
    parser.add_argument("--gaussian_schur", action='store_true',
                        help='If set, uses the schur complement to compute the gaussian covariance matrix.'
                             'If unset, uses matrix multiplications respecting gaussian_pseudo_inverse')
    # batch sizes
    parser.add_argument("--residual_batch", type=int, default=None,
                        help="Number of samples to use in a batch for computing the residual uncertainty")
    parser.add_argument("--empirical_batch", type=int, default=None,
                        help="Number of samples to use in a batch for empirical methods")
    parser.add_argument("--gaussian_batch", type=int, default=100,
                        help="Number of samples to use in a batch for computing the gaussian covariance")
    parser.add_argument("--method_batch", type=int, default=100,
                        help="Number of samples to use in a method batch")

    # general properties
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
    # TODO: this line is the last holdout for importing the starcraft dataset here
    ds = CsvDatasetSplits.load(datasetPath).toTorch()

    # compute residual, it is just a function of regressor and dataset so only need one
    residual = Tensor([0])
    if args.residual_batch is not None:
        startTime = perf_counter()
        residual = estimateResidual(regressor, DataLoader(ds.validate, shuffle=False, batch_size=args.residual_batch))
        endTime = perf_counter()
        logging.info(f"Computed residual uncertainty of {residual}. Took {endTime - startTime}")
    else:
        logging.info(f"Skipping computing residual, set residual_batch to use residual.")

    # setup experiments
    methods: List[Method] = []
    rand = Generator()
    rand.manual_seed(args.seed)

    # create data loader for empirical method
    empiricalLoader: Optional[DataLoader] = None
    if args.empirical_batch is not None:
        empiricalLoader = DataLoader(ds.validate, shuffle=False, batch_size=args.empirical_batch)

    # learn gaussian distribution, TODO: consider saving this per dataset as it will take awhile for StarcraftImage
    # TODO: should this be optional?
    logging.info("Learning gaussian distribution")
    startTime = perf_counter()
    gaussian = ConditionalGaussianDistribution.fromDataloader(
        ds.metadata, DataLoader(ds.train, batch_size=args.gaussian_batch, shuffle=False),
        schur=args.gaussian_schur, leastSquares=not args.gaussian_pseudo_inverse
    )
    endTime = perf_counter()
    logging.info(f"Learned gaussian distribution in {endTime - startTime} seconds")

    # add methods
    def method(method: Method):
        """Adds an method"""
        methods.append(method)

    def imputator(imputator: Imputator):
        """Adds all three basic imputation methods"""
        method(BasicCombinationMethod(regressor, imputator))
        # add empirical if requested
        if empiricalLoader is not None:
            method(EmpiricalUncertaintyByCount(regressor, imputator, ds.metadata, empiricalLoader, residual))
            method(EmpiricalUncertaintyByFeature(regressor, imputator, ds.metadata, empiricalLoader, residual))

    def monteCarlo(distribution: Distribution):
        for samples in args.mc_samples:
            method(MonteCarloMethod(regressor, distribution, samples))

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
        # TODO: can we even do augmented mice with data loaders?
        # imputator(MiceImputator(ds.metadata, iterations, ds.train.features, "Training Features"))

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
        numToDrop = int(totalFeatures*missing)

        # generate two seeds, one for the data loader (since we calculate missing features on the fly and want them
        # consistent across experiments) and one for the experiments (to ensure any random results are consistent
        # regardless of thread count)
        seeds = torch.randint(0, 0x7fffffff, (2,), generator=rand)  # max is just 32-bit signed int max
        for method in methods:
            dropFeatures = DataLoader(FeatureCountRemovingDataset(
                ds.test, ds.metadata, numToDrop, torch.Generator().manual_seed(seeds[0].item())
            ))
            experiments.append(Experiment(method, ds.metadata.name, missingName, missing, residual, data=dropFeatures,
                                          rand=torch.Generator().manual_seed(seeds[1].item()),
                                          storeAllResults=args.write_all_results))

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

                dropFeature = DataLoader(SpecificFeatureRemovingDataset(ds.test, torch.eq(groups, index)),
                                         batch_size=args.method_batch, shuffle=False)
                appendExperiments(experiments, methods, ds.metadata.name, featureName,
                                  residual=residual, rand=rand, storeAllResults=args.write_all_results)

        if args.inverted_feature_impact:
            # create experiment for each feature
            for index in range(totalFeatures):
                featureName = "not " + ds.metadata.featureName(index)
                logging.info(f"Setting up experiments for '{featureName}'")

                dropFeature = DataLoader(SpecificFeatureRemovingDataset(ds.test, torch.ne(groups, index)),
                                         batch_size=args.method_batch, shuffle=False)
                appendExperiments(experiments, methods, ds.metadata.name, featureName,
                                  residual=residual, rand=rand, storeAllResults=args.write_all_results)

    # if -1, give each experiment its own thread
    distributeTasks(experiments, args.threads)
    successful = len([exp for exp in experiments if exp.processedSamples == 0])
    logging.info(f"Finished running {successful}/{len(experiments)} experiments")

    # save all experiment results to the relevant CSV files
    outputName = f"{args.name}-{date}"
    summaryPath = os.path.join(outputFolder, f"{outputName}-summary.csv")
    allPath = os.path.join(outputFolder, f"{outputName}-all.csv")
    logging.info(f"Saving results to {summaryPath}{f' and {allPath}' if args.write_all_results else ''}")
    with open(summaryPath, "w") as summaryFile:
        # summary CSV has one row per experiment
        summaryCsv = csv.writer(summaryFile)

        # if requested, all CSV has one row per sample
        allFile: Optional[TextIO] = None
        allCsv = None
        if args.write_all_results:
            allFile = open(allPath, "w")
            allCsv = csv.writer(allFile)

        # write headers
        Experiment.writeResultHeaders(summaryCsv, allCsv)
        # write rows
        for experiment in experiments:
            experiment.writeResults(summaryCsv, allCsv)
    logging.info("Finished saving results")
