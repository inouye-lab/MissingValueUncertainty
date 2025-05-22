import argparse
import csv
import logging
import os
from typing import List

import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from mvu.dataset.csv import import_from_csv
from mvu.dataset.mutators import SpecificFeatureRemovingDataset
from mvu.explanation.actions import createActionSpace
from mvu.explanation.calibration import MVCEExperiment
from mvu.explanation.moments import MethodOfMomentsDecisionMaker
from mvu.logger import setupLogging
from mvu.model.distribution import GaussianParameters, ConditionalGaussianDistribution
from mvu.model.generator import BatchGenerator, SingleSampleImputator
from mvu.model.imputator import ZeroImputator, Imputator
from mvu.model.method import MonteCarloBatchMethod, BasicCombinationMethod, ScaleMaxBetaVarianceMethod, Method
from mvu.model.regressor import NaiveLinearRegressor
from mvu.threading_utils import distributeTasks
from mvu.util import selectDevice, jsonOrName

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, default="./results/", help='Location to save result CSV')

    # experiment parameters
    parser.add_argument("--threads", type=int, default=-1, help='Number of worker threads to run')
    parser.add_argument("--generator_samples", type=int, nargs='*',
                        help="Monte Carlo samples to take from the generator. If given multiple, adds each.")
    parser.add_argument("--decision_samples", type=int, default=1000,
                        help="Monte Carlo samples to take from the decision distribution. Used for all models")
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")
    parser.add_argument("--dataset_path", type=str, default="./datasets/synthetic/test.csv",
                        help='Location of the input dataset')

    # baseline options
    parser.add_argument("--zero_variance", action='store_true',
                        help="If true, includes imputation baselines at zero variance.")
    parser.add_argument("--beta_variance_scales", type=float, nargs='*', default=[],
                        help="Scales of the beta variance to try for basic imputation.")
    parser.add_argument("--zero_imputation", action='store_true',
                        help="If true, includes zero imputation baseline.")
    parser.add_argument("--single_sample_imputation", action='store_true',
                        help="If true, includes single sample imputation baseline.")
    parser.add_argument("--mean_imputation", action='store_true',
                        help="If true, includes mean imputation baseline.")

    # generator options
    parser.add_argument("--mean_shifts", type=str, nargs='*', default=[],
                        help="Mean shifts to apply to the ground truth generator.")
    parser.add_argument("--flip_variance", action='store_true',
                        help="If set, includes a generator that just swaps the two variances.")
    parser.add_argument("--correlations", type=float, nargs='*', default=[],
                        help="Alternative correlations of the generator to try. Normal is 0.7.")
    parser.add_argument("--covariance_scales", type=float, nargs='*', default=[],
                        help="Alternative correlations of the generator to try. Normal is 1.0.")

    # action space
    parser.add_argument("--action_spaces", nargs='*', type=jsonOrName,
                        help="List of action spaces to consider.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for experiments")
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
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), "synthetic", args=args)

    # setup device
    device = selectDevice(args.cuda_index)
    logging.info(f"Running on {device}")

    torch.manual_seed(args.seed)
    # TODO: does using a generator here make sense?

    # creating classifier
    classifier = NaiveLinearRegressor([1.0, 1.0], -1.0, activation=sigmoid)
    classifier.to(device)
    logging.info(f"Created classifier with weights {classifier.weights} and bias {classifier.bias}")

    # creating generators
    varianceVector = torch.tensor([0.3, 1], device=device)
    gtParams = GaussianParameters.fromVarianceCorrelation(
        torch.zeros((2,), dtype=torch.float, device=device),  # mean
        varianceVector,  # variance
        torch.tensor([[1, 0.7], [0.7, 1]], device=device)  # correlation
    )
    groundTruthGenerator = ConditionalGaussianDistribution(None, gtParams, name="Ground Truth")
    logging.info(f"Running ground truth generator with mean {gtParams.mean} and covariance {gtParams.covariance}")

    # mutated generators
    generators: List[BatchGenerator] = [
        groundTruthGenerator
    ]
    # swap variance of X1 and X2
    if args.flip_variance:
        assert varianceVector[0] != varianceVector[1], "Variance is equivalent flipped"
        covariance = torch.flip(gtParams.covariance, dims=(0, 1))
        generators.append(ConditionalGaussianDistribution(None, GaussianParameters(
            gtParams.mean, covariance
        ), name="Swapped Variances"))
        logging.info(f"Adding swapped variances generator with {gtParams.mean} and covariance {covariance}")
    # changing the correlation
    for correlation in args.correlations:
        assert 1 > correlation > -1, "Correlation must be between -1 and 1"
        params = GaussianParameters.fromVarianceCorrelation(
            gtParams.mean, varianceVector, torch.tensor([[1, correlation], [correlation, 1]], device=device)
        )
        generators.append(ConditionalGaussianDistribution(None, params, name=f"{correlation} Correlation"))
        logging.info(f"Adding {correlation} generator with {gtParams.mean} and covariance {params.covariance}")
    # changing the variance
    for scale in args.covariance_scales:
        assert scale > 0, "Covariance scale must be greater than 0"
        covariance = gtParams.covariance * scale
        generators.append(ConditionalGaussianDistribution(None, GaussianParameters(
           gtParams.mean, covariance
        ), name=f"{scale} * Covariance"))
        logging.info(f"Adding {scale} scaled generator with {gtParams.mean} and covariance {covariance}")
    for shiftStr in args.mean_shifts:
        shift = [float(n) for n in shiftStr.split(',')]
        assert len(shift) == 2, f"Mean shift {shiftStr} has wrong number of elements"
        mean = gtParams.mean + torch.tensor(shift, device=device)
        generators.append(ConditionalGaussianDistribution(None, GaussianParameters(
            mean, gtParams.covariance
        ), name=f"{shiftStr} Mean Shift"))
        logging.info(f"Adding {shift} mean shift generator with {mean} and covariance {gtParams.covariance}")

    # build final method list
    methods: List[Method] = [
        MonteCarloBatchMethod(classifier, generator, samples)
        for samples in args.generator_samples
        for generator in generators
    ]
    if args.zero_variance or len(args.beta_variance_scales) > 0:
        imputators: List[Imputator] = []

        # add zero imputation
        if args.zero_imputation:
            logging.info("Including baseline imputators with zero imputation.")
            imputators.append(ZeroImputator())
        # single sample for symmetry with dataset
        if args.single_sample_imputation:
            logging.info("Including baseline imputators with single sample imputation.")
            imputators.append(SingleSampleImputator(groundTruthGenerator))
        # mean imputation for best results
        if args.mean_imputation:
            logging.info("Including baseline imputators with mean imputation.")
            imputators.append(groundTruthGenerator)

        # log info about variance methods
        if args.zero_variance:
            logging.info(f"Running all baseline imputators zero variance.")
        if len(args.beta_variance_scales) > 0:
            logging.info(
                f"Running all baseline imputators with {args.beta_variance_scales} scaled beta max variance.")

        # for each, do a basic combination method
        for imputator in imputators:
            if args.zero_variance:
                methods.append(BasicCombinationMethod(classifier, imputator))
            for scale in args.beta_variance_scales:
                methods.append(ScaleMaxBetaVarianceMethod(classifier, imputator, scale))

    # setup datasets
    dsClean = import_from_csv(
        "synthetic", args.dataset_path,
        targetFeature="label",
        numericFeatures=["x1", "x2"],
        categoricalFeatures=[]
    ).toTorch()
    dsX1Missing = SpecificFeatureRemovingDataset(dsClean, torch.tensor([True, False]))
    dsX2Missing = SpecificFeatureRemovingDataset(dsClean, torch.tensor([False, True]))
    logging.info(f"Using imported dataset from {args.dataset_path} with {len(dsClean)} samples")

    loaderClean = DataLoader(dsClean, batch_size=args.batch_size, pin_memory=True)
    loaderX1Missing = DataLoader(dsX1Missing, batch_size=args.batch_size, pin_memory=True)
    loaderX2Missing = DataLoader(dsX2Missing, batch_size=args.batch_size, pin_memory=True)

    # map methods to decision makers
    decisionMakers = [MethodOfMomentsDecisionMaker(method, args.decision_samples) for method in methods]

    # finally, build experiment list
    experiments: List[MVCEExperiment] = []
    for actionParams in args.action_spaces:
        logging.info(f"Considering action space {actionParams['name']}")
        lossFunction, actions = createActionSpace(size=1, device=device, **actionParams)
        for decisionMaker in decisionMakers:
            common = dict(
                decisionMaker=decisionMaker,
                actionName=actionParams['name'], lossFunction=lossFunction, actions=actions,
                buckets=args.buckets, trials=args.trials,
                classifier=classifier, device=device
            )
            experiments.append(MVCEExperiment(loaderClean, "Missing X1", loaderX1Missing, **common))
            experiments.append(MVCEExperiment(loaderClean, "Missing X2", loaderX2Missing, **common))

    # get the work started
    distributeTasks(experiments, args.threads)
    finished = [exp for exp in experiments if exp.time is not None]
    logging.info(f"Finished running {len(finished)}/{len(experiments)} experiments.")

    # save all experiment results to the relevant CSV files
    outputName = f"synthetic-{date}"
    csvPath = os.path.join(outputFolder, f"{outputName}.csv")
    logging.info(f"Saving results to {csvPath}")
    with open(csvPath, "w") as csvFile:
        # summary CSV has one row per experiment
        csvWriter = csv.writer(csvFile)

        # write headers
        MVCEExperiment.writeResultHeaders(csvWriter, args.trials)
        # write rows
        for experiment in experiments:
            experiment.writeResults(csvWriter)
    logging.info("Finished saving results")
