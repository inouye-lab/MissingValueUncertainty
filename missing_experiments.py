import argparse
import csv
import json
import logging
import os
from time import perf_counter
from typing import List, Optional, TextIO

import torch
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset

from mvu.dataset.loader import getDatasetSplits
from mvu.model.distribution import ConditionalGaussianDistribution, Distribution, MarginalGaussianDistribution, \
    GaussianParameters
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
    parser.add_argument("--dataset", type=json.loads, default=dict(), help='Path to processed dataset to load')
    parser.add_argument("--regressor", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--regressor_feature", type=int, default=-1,
                        help='Feature index from the regressor to use, if -1 uses all features')
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')
    parser.add_argument("--write_all_results", action='store_true',
                        help="If set, writes a CSV with results from all samples.")

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--mc_samples", type=int, nargs='*', default=[],
                        help="Number of Monte Carlo samples to take")
    parser.add_argument("--mice_iterations", type=int, nargs='*', default=[],
                        help="Number of mice iterations to run")
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")

    # experiment selection
    parser.add_argument("--missing", type=float, default=[], nargs='*',
                        help="Percent of data to treat as missing. If undefined, runs no missing percent experiments")
    parser.add_argument("--feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature separately missing.')
    parser.add_argument("--inverted_feature_impact", action='store_true',
                        help='If set, runs the feature impact experiments by making each feature only present.')

    # method configuration
    parser.add_argument("--skip_basic_imputation", action='store_true',
                        help="If set, basic imputation (non-empirical) will not be run.")
    parser.add_argument("--gaussian_path", type=str, default=None, help='Path to the pretrained gaussian to load')
    parser.add_argument("--gaussian_pseudo_inverse", action='store_true',
                        help='If set, uses the pseudo-inverse for multiplications for the gaussian methods.'
                             'If unset, uses the least squares approach.')
    parser.add_argument("--gaussian_schur", action='store_true',
                        help='If set, uses the schur complement to compute the gaussian covariance matrix.'
                             'If unset, uses matrix multiplications respecting gaussian_pseudo_inverse')
    parser.add_argument("--gaussian_batch", type=int, default=100,
                        help="Number of samples to use in a batch for computing the gaussian covariance")
    parser.add_argument("--gaussian_force_numpy", action='store_true',
                        help="Forces use of numpy to sample gaussians. This may be slower as it does not support GPU, "
                             "but if too many samples are unable to use Torch, skipping the try/catch is faster.")

    # batch sizes
    parser.add_argument("--residual_batch", type=int, default=None,
                        help="Number of samples to use in a batch for computing the residual uncertainty")
    parser.add_argument("--empirical_batch", type=int, default=None,
                        help="Number of samples to use in a batch for empirical methods")
    parser.add_argument("--empirical_limit", type=int, default=None,
                        help="Max number of samples from the validation dataset to use in empirical feature. "
                             "Set to 0 to disable empirical feature. Does not affect empirical by count.")
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
    regressor.setFeatureIndex(args.regressor_feature)

    # device setup
    if args.cuda_index >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", index=args.cuda_index)
        logging.info(f"Using {device} for tensor calculations")
        # PyTorch lazy loads some of its modules which causes issues when in both GPU and threading if we happen to
        # try and load it on multiple threads at the same time. Workaround by using it before we dispatch.
        # see https://github.com/pytorch/pytorch/issues/90613 for more info
        torch.inverse(torch.ones((1, 1), device=device))
    else:
        device = torch.device("cpu")
        # we log whether CUDA is available to make it more clear if it was not an option or force disabled
        logging.info(f"Using {device} for tensor calculations, cuda available: {torch.cuda.is_available()}")
    regressor.to(device)

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)
    if ds.metadata.groups is not None:
        ds.metadata.groups = ds.metadata.groups.to(device)
    logging.info(f"Loaded in dataset {args.name} with {len(ds.train)} training samples, "
                 f"{len(ds.validate)} validation samples, and {len(ds.test)} test samples.")

    # compute residual, it is just a function of regressor and dataset so only need one
    residual = torch.tensor([0], device=device, dtype=torch.float)
    if args.residual_batch is not None:
        startTime = perf_counter()
        residual = estimateResidual(regressor, DataLoader(
            ds.validate, shuffle=False, batch_size=args.residual_batch, pin_memory=True
        ), device=device)
        endTime = perf_counter()
        logging.info(f"Computed residual uncertainty of {residual.cpu()}. Took {endTime - startTime}")
    else:
        logging.info(f"Skipping computing residual, set residual_batch to use residual.")

    # setup experiments
    methods: List[Method] = []
    torch.manual_seed(args.seed)
    rand = Generator()
    rand.manual_seed(args.seed)

    # create data loader for empirical method
    empiricalLoader: Optional[DataLoader] = None
    empiricalFeatureLoader: Optional[DataLoader] = None
    if args.empirical_batch is not None:
        empiricalLoader = DataLoader(ds.validate, shuffle=False, batch_size=args.empirical_batch, pin_memory=True)
        # empirical by feature can be disabled or set to a smaller sample set as its more expensive to run
        if args.empirical_limit is None:
            logging.info("Using all samples for empirical by feature")
            empiricalFeatureLoader = empiricalLoader
        elif args.empirical_limit > 0:
            dataset: Optional[Dataset] = None
            # if we have one batch, just cache it directly as a tensor dataset to speed things up (helps starcraft)
            if args.empirical_limit <= args.empirical_batch:
                logging.info(f"Limiting empirical by feature to {args.empirical_batch} samples, caching the batch")
                for (features, targets) in empiricalLoader:
                    dataset = TensorDataset(features[0:args.empirical_limit, :], targets[0:args.empirical_limit])
                    break
                else:
                    logging.error("Failed to create empirical by feature loader as the validation dataset is empty")
            else:
                logging.info(f"Limiting empirical by feature to {args.empirical_batch} samples")
                dataset = Subset(ds.validate, range(0, args.empirical_limit))
            if dataset is not None:
                empiricalFeatureLoader = DataLoader(dataset, shuffle=False, batch_size=args.empirical_batch,
                                                    pin_memory=True)
        else:
            logging.info("Disabling empirical by feature")

    # learn gaussian distribution
    gaussianParams: GaussianParameters
    if args.gaussian_path is not None:
        logging.info(f"Loading gaussian params from {args.gaussian_path}")
        gaussianParams = GaussianParameters.load(args.gaussian_path).to(device)
    else:
        # TODO: should this be optional?
        logging.info("Learning gaussian distribution")
        startTime = perf_counter()
        gaussianParams = GaussianParameters.fromDataloader(
            ds.metadata.numInputs,
            DataLoader(ds.train, batch_size=args.gaussian_batch, shuffle=False, pin_memory=True),
            device=device
        )
        endTime = perf_counter()
        logging.info(f"Learned gaussian distribution in {endTime - startTime} seconds")

    # add methods
    def method(method: Method):
        """Adds a method"""
        methods.append(method)

    def imputator(imputator: Imputator):
        """Adds all three basic imputation methods"""
        if not args.skip_basic_imputation:
            method(BasicCombinationMethod(regressor, imputator))
        # add empirical if requested
        if empiricalLoader is not None:
            method(EmpiricalUncertaintyByCount(regressor, imputator, ds.metadata, empiricalLoader, residual))
        if empiricalFeatureLoader is not None:
            method(EmpiricalUncertaintyByFeature(regressor, imputator, ds.metadata, empiricalFeatureLoader, residual))

    def monteCarlo(distribution: Distribution):
        for samples in args.mc_samples:
            method(MonteCarloMethod(regressor, distribution, samples))

    # will be using conditional gaussian in several places
    condGaussian = ConditionalGaussianDistribution(
        ds.metadata, gaussianParams,
        schur=args.gaussian_schur,
        leastSquares=not args.gaussian_pseudo_inverse,
        forceNumpy=args.gaussian_force_numpy
    )
    # basic
    imputator(ZeroImputator())
    imputator(ConstantImputator(ds.metadata.normalizeFeatures(gaussianParams.mean), "Mean"))
    imputator(condGaussian)  # Gaussian Conditional Mean Imputation
    # monte carlo
    monteCarlo(MarginalGaussianDistribution(ds.metadata, gaussianParams, forceNumpy=args.gaussian_force_numpy))
    monteCarlo(condGaussian)
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
            # TODO: per method batch size?
            dropFeatures = DataLoader(FeatureCountRemovingDataset(
                ds.test, ds.metadata, numToDrop, torch.Generator().manual_seed(seeds[0].item())
            ), batch_size=args.method_batch, shuffle=False, pin_memory=True)
            experiments.append(Experiment(method, ds.metadata.name, missingName, missing, residual, data=dropFeatures,
                                          rand=torch.Generator().manual_seed(seeds[1].item()), device=device,
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
                                         batch_size=args.method_batch, shuffle=False, pin_memory=True)
                appendExperiments(experiments, methods, ds.metadata.name, featureName,
                                  residual=residual, rand=rand, storeAllResults=args.write_all_results)

        if args.inverted_feature_impact:
            # create experiment for each feature
            for index in range(totalFeatures):
                featureName = "not " + ds.metadata.featureName(index)
                logging.info(f"Setting up experiments for '{featureName}'")

                dropFeature = DataLoader(SpecificFeatureRemovingDataset(ds.test, torch.ne(groups, index)),
                                         batch_size=args.method_batch, shuffle=False, pin_memory=True)
                appendExperiments(experiments, methods, ds.metadata.name, featureName,
                                  residual=residual, rand=rand, storeAllResults=args.write_all_results)

    # if -1, give each experiment its own thread
    distributeTasks(experiments, args.threads)
    finished = [exp for exp in experiments if exp.processedSamples > 0]
    completed = len([exp for exp in finished if exp.processedSamples == exp.totalSamples])
    logging.info(f"Finished running {len(finished)}/{len(experiments)} experiments, "
                 f"with {completed}/{len(finished)} processing all samples.")

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
