import argparse
import csv
import json
import logging
import os
from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from mvu.dataset.loader import getDatasetSplits
from mvu.dataset.mutators import SpecificFeatureRemovingDataset, createMask, IncludeMask, randomDropping
from mvu.dataset.specialized.celeba import CelebADataset
from mvu.explanation.actions import createActionSpaceExpectation
from mvu.explanation.calibration import CalibrationScaleExperiment
from mvu.explanation.decision import DecisionMaker, DiscardingMaskDecisionMaker, ScaleProbabilityDecisionMaker
from mvu.explanation.dirichlet import DirichletDecisionMaker, DirichletClassifier
from mvu.explanation.moments import MethodOfMomentsDecisionMaker
from mvu.logger import setupLogging
from mvu.model.generator import CachingBatchGenerator, BatchMeanImputator, BatchGenerator
from mvu.model.imputator import ZeroImputator, Imputator, SerializableImputator
from mvu.model.method import MonteCarloBatchMethod, BasicCombinationMethod, ScaleMaxBetaVarianceMethod, Method, \
    DiscardingMaskMethod
from mvu.model.regressor import Regressor, NeuralNetworkRegressor
from mvu.model.specialized.resnet import Resnet18Dirichlet
from mvu.threading_utils import distributeTasks
from mvu.util import selectDevice, jsonOrName

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=json.loads, default=dict(), help='Dataset arguments')
    parser.add_argument("--cache_directory", type=str, default=None, help='Location to build the cache')
    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')

    parser.add_argument("--classifier", type=str, help='Path to the pretrained regressor to load')
    parser.add_argument("--dmv_classifier",  action='store_true',
                        help='If set, treats the classifier outputs as alpha values instead of probabilities, using the DMV approximation')
    parser.add_argument("--classifier_feature", type=str, default=None,
                        help='Feature index from the regressor to use, if -1 uses all features')

    # mutator
    parser.add_argument("--mask", type=jsonOrName, default=None, help="Name of the mask to use")
    parser.add_argument("--drop", type=jsonOrName, help="Drop method to use if no mask")

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
    parser.add_argument("--imputators", type=str, nargs='*', default=[],
                        help="If set, adds additional imputators loaded from the specified paths.")
    parser.add_argument("--zero_variance", action='store_true',
                        help="If true, includes zero variance.")
    parser.add_argument("--beta_variance_scales", type=float, nargs='*', default=[],
                        help="Scales of the beta variance to try for basic imputation.")
    parser.add_argument("--probability_scales", type=float, nargs='*', default=[],
                        help="Scales of the probability to produce alpha values for basic imputation.")

    # calibration options
    parser.add_argument("--calibration_scales", type=float, nargs='*', default=[],
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
    classCount = len(ds.metadata.target)
    logging.info(f"Using dataset {args.name} with {len(ds.test)} test samples and {classCount} classes")
    if args.classifier_feature is not None:
        # TODO: generalize this code so other datasets can get original names
        assert isinstance(ds.test, CelebADataset)
        classifier.setFeatureIndex(ds.test.attributes.originalNames.index(args.classifier_feature))
        classCount = 1

    # determine mask
    mask: Optional[Tensor] = None
    if args.mask is not None:
        logging.info("Loading mask " + args.mask["name"])
        mask = createMask(ds.metadata, **args.mask)

    # start setup for decision makers
    decisionMakers: List[DecisionMaker] = []

    # if we have a dirichlet classifier, add the dirichlet decision maker
    includeMask: IncludeMask = IncludeMask.NONE
    if isinstance(classifier, NeuralNetworkRegressor):
        if args.dmv_classifier:
            logging.info(f"Including Dirichlet decision maker")
            classifier.activation = None
            decisionMakers.extend(DirichletDecisionMaker(classifier, args.decision_samples, scale=scale) for scale in args.calibration_scales)
            # substitute the classifier for the remaining methods with one that can handle missing masks
            classifier = DirichletClassifier.fromRegressor(classifier, num_classes=classCount, expected_mask_size=ds.metadata.channels + 1)
        elif classCount == 1:
            logging.info(f"Setting classifier activation function to sigmoid for single class")
            classifier.activation = nn.Sigmoid()
        else:
            logging.info(f"Setting classifier activation function to softmax for multiclass")
            # TODO: will it always be true that we wish to set the activation function like this? maybe it should be set at a nn level
            classifier.activation = nn.Softmax(dim=1)

        # set the proper mask type
        if isinstance(classifier.nn, Resnet18Dirichlet):
            includeMask = IncludeMask.MISSING

    # methods
    methods: List[Method] = []
    # add generator method if we have a caching batch generator
    generator: Optional[BatchGenerator] = None
    if args.cache_directory is not None:
        if mask is None:
            logging.error("Attempting to use a cache directory with no mask, this does not work")
        else:
            logging.info(f"Creating generator using cache at {args.cache_directory}")
            generator = CachingBatchGenerator(None, args.cache_directory, mask.to(device))
            methods.extend(
                MonteCarloBatchMethod(classifier, generator, samples)
                for samples in args.generator_samples
            )

    # basic imputation
    if args.zero_variance or len(args.beta_variance_scales) > 0:
        imputators: List[Imputator] = []

        # add zero imputation
        if args.zero_imputation:
            logging.info("Including baseline imputators with zero imputation.")
            imputators.append(ZeroImputator())

        # add serialized imputators
        for path in args.imputators:
            logging.info(f"Loading imputator from {path}")
            imputator = SerializableImputator.load(path)
            logging.info(f"Found {imputator.name}")
            imputator.to(device)
            imputators.append(imputator)

        # add batch imputator if requested
        if generator is not None and len(args.batch_mean_imputation) > 0:
            logging.info(f"Including batch mean imputators with sizes {args.batch_mean_imputation}.")
            imputators.extend(BatchMeanImputator(generator, size) for size in args.batch_mean_imputation)

        # log info about variance methods
        if args.zero_variance:
            logging.info(f"Running all baseline imputators zero variance.")
        if len(args.beta_variance_scales) > 0:
            logging.info(f"Running all baseline imputators with {args.beta_variance_scales} scaled beta max variance.")

        # for each, do a basic combination method
        for imputator in imputators:
            if args.zero_variance:
                methods.append(BasicCombinationMethod(classifier, imputator))
            for scale in args.beta_variance_scales:
                methods.append(ScaleMaxBetaVarianceMethod(classifier, imputator, scale))

        # add methods for probability scales
        if len(args.probability_scales) > 0:
            if args.dmv_classifier:
                logging.info(f"Adding {len(imputators)} imputators with discarded masks for probability scales {args.probability_scales}.")
                # if we have a mask, strip it from all methods
                maskKeep = torch.arange(0, ds.metadata.channels, device=device)
                decisionMakers.extend(DiscardingMaskDecisionMaker(ScaleProbabilityDecisionMaker(classifier, imputator, args.decision_samples, scale), maskKeep)
                                      for scale in args.probability_scales for imputator in imputators)
            else:
                logging.info(f"Adding {len(imputators)} for probability scales {args.probability_scales}.")
                decisionMakers.extend(ScaleProbabilityDecisionMaker(classifier, imputator, args.decision_samples, scale)
                                      for scale in args.probability_scales for imputator in imputators)

    # map all additional methods to decision makers
    if args.dmv_classifier and includeMask != IncludeMask.NONE:
        logging.info(f"Adding {len(methods)} methods with discarded masks.")
        # if we have a mask, strip it from all methods
        maskKeep = torch.arange(0, ds.metadata.channels, device=device)
        decisionMakers.extend(MethodOfMomentsDecisionMaker(DiscardingMaskMethod(method, maskKeep), args.decision_samples, scale=scale_val) for scale_val in args.calibration_scales for method in methods)
    else:
        logging.info(f"Adding {len(methods)} methods.")
        decisionMakers.extend(MethodOfMomentsDecisionMaker(method, args.decision_samples, scale=scale_val) for scale_val in args.calibration_scales for method in methods)

    # setup datasets
    # if we have any methods beyond the Dirichlet, then use nan for the missing value. 0 is faster but isn't what most methods support
    missingArgs = dict(includeMask=includeMask, missingValue=torch.nan if len(methods) > 0 else 0)
    maskName: str
    if mask is not None:
        logging.info(f"Using masked dataset with mask {args.mask}")
        maskName = args.mask["name"]
        dsMissing = SpecificFeatureRemovingDataset(ds.validate, mask, **missingArgs)
    else:
        logging.info(f"Using randon dropping with arguments {args.drop}")
        maskName = args.drop["name"] # TODO: use dataset name instead
        dsMissing = randomDropping(ds.validate, ds.metadata, **missingArgs, **args.drop)

    # TODO: our newer datasets support returning the original if prompted, don't need loaderClean
    loaderClean = DataLoader(ds.validate, batch_size=args.batch_size, pin_memory=True)
    loaderMissing = DataLoader(dsMissing, batch_size=args.batch_size, pin_memory=True)

    # finally, build experiment list
    experiments: List[CalibrationScaleExperiment] = []

    lossFunctionBatch, actions = createActionSpaceExpectation(args.action_spaces, size=classCount, device=device)
    for decisionMaker in decisionMakers:
        experiments.append(CalibrationScaleExperiment(
            loaderClean, maskName, loaderMissing,
            decisionMaker=decisionMaker,
            lossFunctions=lossFunctionBatch, actions=actions,
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
        CalibrationScaleExperiment.writeResultHeaders(csvWriter, args.trials)
        # write rows
        for experiment in experiments:
            experiment.writeResults(csvWriter)
    logging.info("Finished saving results")
