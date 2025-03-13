import argparse
import json
import logging
import os
from time import perf_counter

import torch
from torch import Generator
from torch.utils.data import Dataset

from mvu.dataset.loader import getDatasetSplits
from mvu.dataset.mutators import createMask, SpecificFeatureRemovingDataset
from mvu.logger import setupLogging
from mvu.model.generator import CachingBatchGenerator
from mvu.model.loader import createBatchGenerator
from mvu.util import jsonOrName, selectDevice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=json.loads, default=dict(), help='Dataset arguments')
    parser.add_argument("--split", type=str, default='test', help='Dataset split to use')
    parser.add_argument("--generator", type=jsonOrName, help='Generator to use')
    parser.add_argument("--cache_directory", type=str, help='Location to build the cache')

    parser.add_argument("--samples", type=int,
                        help="Number of samples to cache in each batch")
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")
    parser.add_argument("--mask", type=jsonOrName, help="Name of the mask to use")

    # general properties
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.cache_directory
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args=args)
    logging.info(f"Starting to build caches for {args.name}")

    # seed random parameters
    torch.manual_seed(args.seed)  # TODO: anymore work for seeds?
    rand = Generator()
    rand.manual_seed(args.seed)

    # device setup
    device = selectDevice(args.cuda_index)

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)

    logging.info("Loading mask")
    mask = createMask(ds.metadata, **args.mask)
    dataset: Dataset
    if args.split == 'test':
        dataset = ds.test
    elif args.split == 'train':
        dataset = ds.train
    elif args.split == 'validation':
        dataset = ds.validate
    else:
        raise ValueError(f"Unknown dataset split {args.split}")
    logging.info(f"Using {args.split} dataset split")
    withMissing = SpecificFeatureRemovingDataset(dataset, mask)

    logging.info("Constructing generator")
    baseGenerator = createBatchGenerator(device=device, **args.generator)
    generator = CachingBatchGenerator(baseGenerator, args.cache_directory, mask.to(device))

    # generate the samples
    logging.info("Generating batches")

    # no need for a data loader, we are processing them one at a time without shuffling
    for (features, labels, index) in withMissing:
        logging.info(f"Generating batch for sample index {index}")
        batchStart = perf_counter()
        batch = generator.createBatch(features.to(device), args.samples, int(index), rand)
        logging.info(f"Generated batch of shape {batch.shape} for index {index} in "
                     f"{perf_counter() - batchStart:.5f} seconds")
