import argparse
import csv
import logging
import os
import sys
import pandas as pd
from typing import List, Optional, Tuple, Dict

import torch
from torch import Tensor
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader


from mvu.dataset.csv import import_from_csv
from mvu.dataset.mutators import SpecificFeatureRemovingDataset
from mvu.explanation.actions import createActionSpace
from mvu.explanation.calibration import CalibrationExperiment, CalibrationScaleExperiment
from mvu.explanation.moments import MethodOfMomentsDecisionMaker
from mvu.logger import setupLogging
from mvu.model.distribution import GaussianParameters, ConditionalGaussianDistribution
from mvu.model.generator import BatchGenerator, SingleSampleImputator
from mvu.model.imputator import ZeroImputator
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
    parser.add_argument("--calibration_path", type=str, default="./datasets/synthetic/calibration.csv",
                        help='Location of the calibration data')

    # baseline options
    parser.add_argument("--imputation_baselines", action='store_true',
                        help="If true, includes baselines for basic types of imputation.")
    parser.add_argument("--zero_imputation", action='store_true',
                        help="If true, includes baselines for basic types of imputation.")
    parser.add_argument("--singleSample_imputation", action='store_true',
                        help="If true, includes baselines for basic types of imputation.")
    parser.add_argument("--groundTruthGenerator_imputation", action='store_true',
                        help="If true, includes baselines for basic types of imputation.")

    parser.add_argument("--beta_variance_scales", type=float, nargs='*', default=[],
                        help="Scales of the beta variance to try for basic imputation.")

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
    # Below are techniques to develop new generators
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
    if args.zero_imputation or args.singleSample_imputation or args.groundTruthGenerator_imputation or len(args.beta_variance_scales) > 0:
        logging.info("Including baseline imputators with zero imputation, single sample imputation, and "
                     "conditional gaussian. Running all imputators with zero variance.")
        for scale in args.beta_variance_scales:
            logging.info(f"Running all baseline imputators with {scale} scaled beta max variance.")
        imputators = []
        if args.zero_imputation:
            imputators.append(ZeroImputator())
        if args.singleSample_imputation:
            imputators.append(SingleSampleImputator(groundTruthGenerator))
        if args.groundTruthGenerator_imputation:
            imputators.append(groundTruthGenerator)

        for imputator in imputators:
            if args.zero_imputation or args.singleSample_imputation or args.groundTruthGenerator_imputation:
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
    # k = [0.01,0.1,1,10,100]
    #k = [0.1, 0.5, 1, 5, 10]
    # k = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]
    scale_val = [1, 1, 0.75, 0.25, 5, 1, 2.5]
    for index, m in enumerate(methods):
        print(f'The method is: {m.name} and scale:{scale_val[index]}')
    # sys.exit()

    df_method_scale = pd.read_csv(args.calibration_path)
    df_method_scale['Scale'] = pd.to_numeric(df_method_scale['Scale'], errors='coerce')
    method_to_scale = dict(zip(df_method_scale['Method'], df_method_scale['Scale']))

    for key, value in method_to_scale.items():
        print(f'The method is: {key.name} and scale:{value}')


    #decisionMakers = [MethodOfMomentsDecisionMaker(method, args.decision_samples, scale=scale_val[index]) for index, method in enumerate(methods)]# This is where scale
    decisionMakers = [MethodOfMomentsDecisionMaker(method, args.decision_samples, scale=method_to_scale[method.name]) for
                      method in methods]  # This is where scale

    # finally, build experiment list
    experiments: List[CalibrationScaleExperiment] = []
    lossFunctionBatch: List[callable] = []
    actions: Optional[Tensor] = None
    for actionParams in args.action_spaces:
        logging.info(f"Considering action space {actionParams['name']}")
        lossFunction, newActions = createActionSpace(size=1, device=device, **actionParams)
        lossFunctionBatch.append(lossFunction)
        if actions is None:
            actions = newActions
        elif torch.any(torch.ne(actions, newActions)):
            # compare actions and new actions, error (exit(0)) or throw if they mismatch
            pass
    for decisionMaker in decisionMakers:
        common = dict(
            decisionMaker=decisionMaker,
            lossFunctions=lossFunctionBatch, actions=actions,
            buckets=args.buckets, trials=args.trials,
            classifier=classifier, device=device,
            scale=decisionMaker.scale
        )
        experiments.append(CalibrationScaleExperiment(loaderClean, "Missing X1", loaderX1Missing, **common))
        experiments.append(CalibrationScaleExperiment(loaderClean, "Missing X2", loaderX2Missing, **common))

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
        CalibrationScaleExperiment.writeResultHeaders(csvWriter, args.trials)
        # write rows
        for experiment in experiments:
            experiment.writeResults(csvWriter)

    # # mapping from (methodName, scale) to list of all different mask MVCE
    # #methodScaleErrors: Dict[Tuple[str, float], List[Tensor]] = dict()
    # methodScaleErrors: Dict[str, Dict[float, List[Tensor]]] = dict()
    # for experiment in experiments:
    #     mean = experiment.results.mean()
    #     pass
    # # CSV of method, scale, average MVCE
    #
    # # method name to best k scale and best average MVCE
    # methodScale: Dict[str, Tuple[float, Tensor]]
    # # CSV of method, best scale, average MVCE

    # # Mean MVCE Result by Method
    # df = pd.read_csv(csvPath)
    # avg_by_method = (
    #     df
    #     .groupby('Method')['MVCE Mean']
    #     .mean()
    #     .reset_index(name='Avg_MVCE_Mean')
    # )
    #
    # outputName_1 = f"synthetic-{date}_1"
    # csvPath_1 = os.path.join(outputFolder, f"{outputName_1}.csv")
    #
    # avg_by_method.to_csv(csvPath_1, index=False)
    #
    # # Method and K seprated For plotting
    #
    # df_1 = pd.read_csv(csvPath_1)
    #
    # df_split = df_1['Method'].str.split('_', n=1, expand=True)
    #
    # df_1['Method'] = df_split[0]  # part before the underscore
    # df_1['k'] = df_split[1]  # part after the underscore
    #
    # # (Optional) If you know k is numeric, convert its dtype
    # df_1['k'] = pd.to_numeric(df['k'], errors='ignore')
    #
    #
    #
    # outputName_2 = f"synthetic-{date}_2"
    # csvPath_2 = os.path.join(outputFolder, f"{outputName_2}.csv")
    #
    # df_1.to_csv(csvPath_2, index=False)
    #
    # # Best K for a method
    # df_2 = pd.read_csv(csvPath_2)
    # df_2['k'] = pd.to_numeric(df_2['k'], errors='ignore')
    # idx_min = df_2.groupby('Method')['Avg_MVCE_Mean'].idxmin()
    # best_k_df = df_2.loc[idx_min, ['Method', 'k', 'Avg_MVCE_Mean']].reset_index(drop=True)
    #
    # outputName_3 = f"synthetic-{date}_3"
    # csvPath_3 = os.path.join(outputFolder, f"{outputName_3}.csv")
    #
    # best_k_df.to_csv(csvPath_2, index=False)


    logging.info("Finished saving results")
