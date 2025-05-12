import argparse
import logging
import os
from typing import List

import pandas as pd
import torch
from torch import Generator
from torch.nn.functional import sigmoid

from mvu.logger import setupLogging
from mvu.model.distribution import GaussianParameters, ConditionalGaussianDistribution
from mvu.model.regressor import NaiveLinearRegressor
from mvu.util import selectDevice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default=None, help='Output CSV name')
    parser.add_argument("--output", type=str, default="./datasets/synthetic/", help='Location to save result dataset')
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of samples to include in the dataset")
    parser.add_argument("--cuda_index", type=int, default=-1,
                        help="Index to use for CUDA, set to -1 to force CPU")

    # Missing setup
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)

    # setup device
    device = selectDevice(args.cuda_index)
    torch.manual_seed(args.seed)
    rand = Generator()
    rand.manual_seed(args.seed)

    # TODO: any dataset parameters?
    # creating classifier
    classifier = NaiveLinearRegressor([1.0, 1.0], -1.0, activation=sigmoid)
    classifier.to(device)
    logging.info(f"Created classifier with weights {classifier.weights} and bias {classifier.bias}")

    # creating generators
    mean = torch.zeros((2,), dtype=torch.float, device=device)
    varianceVector = torch.tensor([0.3, 1], device=device)
    gtParams = GaussianParameters.fromVarianceCorrelation(
        mean,  # mean
        varianceVector,  # variance
        torch.tensor([[1, 0.7], [0.7, 1]], device=device)  # correlation
    )
    generator = ConditionalGaussianDistribution(None, gtParams, name="Ground Truth")
    logging.info(f"Running ground truth generator with mean {gtParams.mean} and covariance {gtParams.covariance}")

    # generate dataset
    xValues = generator.createBatch(mean * torch.nan, args.samples, rand=rand)
    phiValues = classifier.predict(xValues)
    yValues = torch.bernoulli(phiValues)
    labels: List[str] = []
    for i in range(mean.shape[0]):
        labels.append(f"x{i+1}")
    labels.append("label")
    labels.append("phi")

    # create data frame
    df = pd.DataFrame(torch.cat((xValues, yValues.unsqueeze(1), phiValues.unsqueeze(1)), dim=1).cpu().numpy(), columns=labels)
    df['label'] = pd.to_numeric(df['label'], errors='raise').astype("Int32")
    # save data frame
    path = os.path.join(outputFolder, f"{date if args.name is None else args.name}.csv")
    df.to_csv(path, index=False)
    logging.info(f"Saved dataset to {path}")
