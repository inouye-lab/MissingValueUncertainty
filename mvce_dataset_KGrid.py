import argparse
import csv
import json
import logging
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from mvu.dataset.loader import getDatasetSplits
from mvu.dataset.mutators import SpecificFeatureRemovingDataset, createMask
from mvu.dataset.specialized.celeba import CelebADataset
from mvu.explanation.actions import createActionSpace
from mvu.explanation.calibration import CalibrationExperiment
from mvu.explanation.moments import MethodOfMomentsDecisionMaker
from mvu.logger import setupLogging
from mvu.model.generator import SingleSampleImputator, CachingBatchGenerator, BatchMeanImputator
from mvu.model.imputator import ZeroImputator
from mvu.model.method import MonteCarloBatchMethod, BasicCombinationMethod, ScaleMaxBetaVarianceMethod, Method
from mvu.model.regressor import Regressor
from mvu.threading_utils import distributeTasks
from mvu.util import selectDevice, jsonOrName

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=json.loads, default=dict(), help='Dataset arguments')
    parser.add_argument("--cache_directory", type=str, help='Location to build the cache')
    parser.add_argument("--mask", type=jsonOrName, help="Name of the mask to use")
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')

    parser.add_argument("--classifier", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--classifier_feature", type=str, default=None,
                        help='Feature index from the regressor to use, if -1 uses all features')

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--generator_samples", type=int, nargs='*', default=[],
                        help="Monte Carlo samples to take from the generator. If given multiple, adds each.")
    parser.add_argument("--decision_samples", type=int, default=1000,
                        help="Monte Carlo samples to take from the decision distribution. Used for all models")
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")

    # baseline options
    parser.add_argument("--zero_imputation", action='store_true',
                        help="If true, includes zero imputation.")
    parser.add_argument("--beta_variance_scales", type=float, nargs='*', default=[],
                        help="Scales of the beta variance to try for basic imputation.")

    # action space
    parser.add_argument("--action_spaces", nargs='*', type=jsonOrName,
                        help="List of action spaces to consider.")
    parser.add_argument("--class_count", type=int, default=1,
                        help="Number of class actions to include in the dataset.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for experiments")
    parser.add_argument("--batch_mean_imputation", nargs='*', type=int, default=[],
                        help="Batch sizes for the generator batch mean method")
    # MVCE
    parser.add_argument('--buckets', type=int, default=10,
                        help='Number of buckets for calibration error calculations')
    parser.add_argument('--trials', type=int, default=10,
                        help='Number of trials to run for statistics on consistency')

    # Missing setup
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)

    # setup device
    device = selectDevice(args.cuda_index)
    logging.info(f"Running on {device}")

    torch.manual_seed(args.seed)
    # TODO: does using a generator here make sense?

    # creating classifier
    classifier = Regressor.load(args.classifier)
    classifier.to(device)

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)
    logging.info(f"Using dataset {args.name} with {len(ds.test)} test samples")
    # TODO: generalize this code so other datasets can get original names
    assert isinstance(ds.test, CelebADataset)
    if args.classifier_feature is not None:
        classifier.setFeatureIndex(ds.test.attributes.originalNames.index(args.classifier_feature))

    logging.info("Loading mask " + args.mask["name"])
    mask = createMask(ds.metadata, **args.mask)
    dsMissing = SpecificFeatureRemovingDataset(ds.test, mask)

    logging.info(f"Creating generator using cache at {args.cache_directory}")
    generator = CachingBatchGenerator(None, args.cache_directory, mask.to(device))

    # methods
    methods: List[Method] = [
        MonteCarloBatchMethod(classifier, generator, samples)
        for samples in args.generator_samples
    ]
    if args.zero_imputation or len(args.beta_variance_scales) > 0:
        logging.info("Including baseline imputators with zero imputation, single sample imputation, and "
                     "conditional gaussian. Running all imputators with zero variance.")
        for scale in args.beta_variance_scales:
            logging.info(f"Running all baseline imputators with {scale} scaled beta max variance.")
        for imputator in [ZeroImputator(), *[BatchMeanImputator(generator, size) for size in args.batch_mean_imputation]]:
            if args.zero_imputation:
                methods.append(BasicCombinationMethod(classifier, imputator))
            for scale in args.beta_variance_scales:
                methods.append(ScaleMaxBetaVarianceMethod(classifier, imputator, scale))

    # setup datasets
    loaderClean = DataLoader(ds.test, batch_size=args.batch_size, pin_memory=True)
    loaderMissing = DataLoader(dsMissing, batch_size=args.batch_size, pin_memory=True)

    k = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]

    # map methods to decision makers
    decisionMakers = [MethodOfMomentsDecisionMaker(method, args.decision_samples, scale=scale_val) for scale_val in k for method in methods]

    # finally, build experiment list
    experiments: List[CalibrationExperiment] = []
    for actionParams in args.action_spaces:
        logging.info(f"Considering action space {actionParams['name']}")
        lossFunction, actions = createActionSpace(size=args.class_count, device=device, **actionParams)
        for decisionMaker in decisionMakers:
            experiments.append(CalibrationExperiment(
                loaderClean, args.mask["name"], loaderMissing,
                decisionMaker=decisionMaker,
                actionName=actionParams['name'], lossFunction=lossFunction, actions=actions,
                buckets=args.buckets, trials=args.trials,
                classifier=classifier, device=device
            ))

    # get the work started
    distributeTasks(experiments, args.threads)
    finished = [exp for exp in experiments if exp.time is not None]
    logging.info(f"Finished running {len(finished)}/{len(experiments)} experiments.")

    # save all experiment results to the relevant CSV files
    outputName = f"{args.name}-{date}"
    csvPath = os.path.join(outputFolder, f"{outputName}.csv")
    logging.info(f"Saving results to {csvPath}")
    with open(csvPath, "w") as csvFile:
        # summary CSV has one row per experiment
        csvWriter = csv.writer(csvFile)

        # write headers
        CalibrationExperiment.writeResultHeaders(csvWriter, args.trials)
        # write rows
        for experiment in experiments:
            experiment.writeResults(csvWriter)
    logging.info("Finished saving results")
