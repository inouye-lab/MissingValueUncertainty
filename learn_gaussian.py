import argparse
import json
import logging
import os
from time import perf_counter

from torch.utils.data import DataLoader

from mvu.dataset.loader import getDatasetSplits, validateArgs
from mvu.logger import setupLogging
from mvu.model.distribution import MarginalGaussianDistribution, GaussianParameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=None, help='Parameters to load the dataset')
    parser.add_argument("--output", type=str, default="./datasets/gaussian/", help='Location to save the result')
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of samples to use in a batch for computing the gaussian covariance")
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    setupLogging(args.verbose, os.path.join(args.output, "log"), args.name)

    # load in dataset
    ds = getDatasetSplits(args.name, **validateArgs(args.dataset))

    # Create gaussian
    logging.info("Learning gaussian distribution")
    startTime = perf_counter()
    gaussian = GaussianParameters.fromDataloader(
        ds.metadata.numInputs, DataLoader(ds.train, batch_size=args.batch_size, shuffle=False), showProgress=True
    )
    endTime = perf_counter()
    logging.info(f"Finished learning gaussian after {endTime - startTime} seconds")

    # Save gaussian
    path = os.path.join(args.output, args.name + ".pklz")
    logging.info(f"Saving gaussian at {path}")
    gaussian.save(path)

