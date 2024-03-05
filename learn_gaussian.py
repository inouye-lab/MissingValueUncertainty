import argparse
import json
import logging
import os
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from mvu.dataset.loader import getDatasetSplits
from mvu.logger import setupLogging
from mvu.model.distribution import GaussianParameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=dict(), help='Parameters to load the dataset')
    parser.add_argument("--output", type=str, default="./datasets/gaussian/", help='Location to save the result')
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of samples to use in a batch for computing the gaussian covariance")
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')
    parser.add_argument("--force_cpu", action='store_true',
                        help="If set, forces using the CPU for calculations instead of the GPU.")

    args = parser.parse_args()

    # start logging
    setupLogging(args.verbose, os.path.join(args.output, "log"), args.name, args)

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)

    # device setup
    device = torch.device("cuda" if not args.force_cpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} for tensor calculations, cuda available: {torch.cuda.is_available()}")

    # Create gaussian
    logging.info("Learning gaussian distribution")
    startTime = perf_counter()
    gaussian = GaussianParameters.fromDataloader(
        ds.metadata.numInputs, DataLoader(ds.train, batch_size=args.batch_size, shuffle=False),
        showProgress=True, device=device
    ).cpu()
    endTime = perf_counter()
    logging.info(f"Finished learning gaussian after {endTime - startTime} seconds")

    # Save gaussian
    path = os.path.join(args.output, args.name + ".pklz")
    logging.info(f"Saving gaussian at {path}")
    gaussian.save(path)

