import argparse
import json
import logging
import os

import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from mvu.dataset import DatasetSplits, INDEX_SAMPLE, Dataset
from mvu.logger import setupLogging
from mvu.regressor import RidgeRegressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("path", type=str, default=None, help='Location of the dataset binary for processing')
    parser.add_argument("--output", type=str, default="./models/ridge/", help='Location of the CSV file to parse')

    parser.add_argument('--params', type=json.loads, default=dict(),
                        help='Parameters for the ridge regression')
    parser.add_argument('--cv_params', type=json.loads, default=None,
                        help='Parameters for cross validation')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Seed for random permutations')

    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')

    args = parser.parse_args()

    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)
    logging.info(f"Starting to train {args.name}")

    # load in dataset
    path = args.path
    if path is None:
        path = f"./datasets/binary/{args.name}.pklz"
    logging.info(f"Loading dataset from {path}")
    ds = DatasetSplits.load(path)

    # construct model
    logging.info("Constructing ridge model")
    model = Ridge(
        random_state=args.seed,
        **args.params
    )
    # use cross validation if requested
    if args.cv_params is not None:
        logging.info("Performing cross validation")
        model = GridSearchCV(model, args.cv_params, cv=min(5, ds.train.targets.shape[INDEX_SAMPLE]), n_jobs=-1)

    # fit model
    logging.info("Starting model fit")
    model.fit(ds.train.features.numpy(), ds.train.targets.numpy())

    # extract the best CV model if used
    if args.cv_params is not None:
        logging.info(f"Best cross validation params: {str(model.best_params_)}")
        model = model.best_estimator_

    # score the model
    regressor = RidgeRegressor(model)

    def evaluate(dataset: Dataset, split: str):
        predicted = regressor.predict(dataset.features)
        mse = torch.mean((predicted - dataset.targets) ** 2)
        logging.info(f"MSE for {split}: {mse}")
    evaluate(ds.train, "train")
    evaluate(ds.validate, "validation")
    evaluate(ds.test, "test")

    # save the model
    # wrapping in RidgeRegressor makes it more convenient to load later
    outputPath = os.path.join(outputFolder, f"{args.name}-{date}.pklz")
    logging.info(f"Saving model to {outputPath}")
    regressor.save(outputPath)
