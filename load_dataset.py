import argparse
import logging
import os

import torch

from mvu.dataset import import_from_csv, split_dataset
from mvu.logger import setupLogging, dumpArgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("path", type=str, default=None, help='Location of the CSV file to parse')
    parser.add_argument("--output", type=str, default="./datasets/binary/", help='Location of the CSV file to parse')

    # Features
    parser.add_argument("--target", type=str, help="Target feature")
    parser.add_argument("--numeric", default=[], nargs='*', type=str, help="Numerical input features")
    parser.add_argument("--categorical", default=[], nargs='*', type=str,
                        help="Categorical or boolean input features")

    # Splits
    parser.add_argument('--validate_percent', type=float, default=0.2,
                        help='Percentage of the dataset for validation')

    parser.add_argument('--test_percent', type=float, default=0.3,
                        help='Percentage of the dataset for testing')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Seed for random permutations')

    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    setupLogging(args.verbose, os.path.join(args.output, "log"), args.name)

    # dump arguments in case we want them for later
    dumpArgs(args, os.path.join(args.output, f"{args.name}-args.json"))

    # Default CSV file to the dataset name if not given
    path = args.path
    if args.path is None:
        path = f"./datasets/csv/{args.name}.csv"

    # load in data from CSV
    unsplit = import_from_csv(args.name, path, args.target, args.numeric, args.categorical)

    # split into train, validate, test
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    split = split_dataset(unsplit, args.validate_percent, args.test_percent, generator)

    # write to binary
    path = os.path.join(args.output, args.name + ".pklz")
    logging.info(f"Saving dataframe at {path}")
    split.save(path)
