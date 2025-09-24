import argparse
import json
import logging
import os

from torch.utils.data import Dataset, DataLoader

from mvu.dataset.loader import getDatasetSplits
from mvu.logger import setupLogging
from mvu.model.imputator import ConstantImputator
from mvu.util import selectDevice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset details
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=dict(), help='Parameters to load the dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during training')
    parser.add_argument('--split', type=str, default="train", help='Dataset split to use')

    # result
    parser.add_argument("--output", type=str, default="./models/mean/", help='Location to save final mean imputator')

    # misc
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)
    logging.info(f"Starting to learn mean for {args.name}")

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)

    # device setup
    device = selectDevice(args.cuda_index)

    # select dataset split
    dataset: Dataset
    if args.split == "test":
        dataset = ds.test
    elif args.split == "train":
        dataset = ds.train
    elif args.split == "validate":
        dataset = ds.validate
    else:
        raise ValueError(f"Unknown split {args.split}")
    logging.info(f"Using dataset split {args.split}")

    # setup data loading
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # learn the mean
    logging.info(f"Started learning {args.name} mean")
    imputator = ConstantImputator.meanFromDataloader(dataLoader, showProgress=True, device=device)

    # save the result
    imputator.constant.cpu()
    outputPath = os.path.join(outputFolder, f"{args.name}.pklz")
    logging.info(f"Saving mean to {outputPath}")
    imputator.save(outputPath)