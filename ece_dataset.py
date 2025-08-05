import argparse
import json
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mvu.dataset.loader import getDatasetSplits
from mvu.explanation.calibration import computeECE
from mvu.explanation.dirichlet import DirichletClassifier
from mvu.logger import setupLogging
from mvu.model.regressor import Regressor, NeuralNetworkRegressor
from mvu.model.specialized.resnet import Resnet18Dirichlet
from mvu.util import selectDevice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("--dataset", type=json.loads, default=dict(), help='Dataset arguments')
    parser.add_argument("--output", type=str, default="./results/ece/", help='Location to save result CSV')

    parser.add_argument("--classifier", type=str, help='Path to the pretrained regressor to load')

    # experiment parameters
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")

    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for experiments")
    # ECE
    parser.add_argument('--buckets', type=int, default=10,
                        help='Number of buckets for calibration error calculations')
    parser.add_argument('--split', type=str, default="test",
                        help='Dataset split to use')

    # Missing setup
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, outputFolder, args.name, args=args)

    # setup device
    device = selectDevice(args.cuda_index)
    logging.info(f"Running on {device}")

    torch.manual_seed(args.seed)


    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)
    classCount = len(ds.metadata.target)
    logging.info(f"Using dataset {args.name} with {len(ds.test)} test samples and {classCount} classes")

    # creating classifier
    classifier = Regressor.load(args.classifier)
    classifier.to(device)
    if isinstance(classifier, NeuralNetworkRegressor):
        if isinstance(classifier.nn, Resnet18Dirichlet):
            logging.info(f"Using dirichlet decision maker")
            # substitute the classifier for the remaining methods with one that convert to mean
            classifier = DirichletClassifier.fromRegressor(classifier, num_classes=classCount)
        elif classCount == 1:
            logging.info(f"Setting classifier activation function to sigmoid for single class")
            classifier.activation = nn.Sigmoid()
        else:
            logging.info(f"Setting classifier activation function to softmax for multiclass")
            # TODO: will it always be true that we wish to set the activation function like this? maybe it should be set at a nn level
            classifier.activation = nn.Softmax(dim=1)

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

    # load in dataset
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    # finally, compute ECE
    computeECE(loader, classifier, classCount, args.buckets, device)
