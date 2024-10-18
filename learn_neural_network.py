import argparse
import copy
import json
import logging
import os
from time import perf_counter

import torch
import sys
from torch import Tensor, Generator
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, random_split, Dataset, DataLoader,ConcatDataset

from mvu.dataset.loader import getDatasetSplits
from mvu.logger import setupLogging
from mvu.model.loader import createRegressorFromJson
from mvu.model.regressor import NeuralNetworkRegressor
from mvu.util import jsonOrString, process_tensor # Adding the extra function here
from mvu.dataset.mutators import createMask, SpecificFeatureRemovingDataset

import wandb


# Set the random seed locally for reproducibility without affecting global seed
def split_dataset(dataset, n):
    dataset_len = len(dataset)
    part_len = dataset_len // n  # Base length of each part

    # Create a list of lengths for each part
    lengths = [part_len] * n  # Initially set all parts to the same length

    # Adjust the last part to account for any remainder if the dataset is not divisible by n
    remainder = dataset_len % n
    for i in range(remainder):
        lengths[i] += 1  # Distribute remainder across the first few parts

    # Use random_split to split the dataset
    subsets = random_split(dataset, lengths)

    return subsets  # Return the n dataset objects

def mask_split_dataset(split_Traindatasets, ds,args):
    split_TransformedTraindatasets = []
    for i, subset in enumerate(split_Traindatasets):
        withMissing = subset
        if args.mask[i] == "top":
            mask = createMask(ds.metadata, "top")
            withMissing = SpecificFeatureRemovingDataset(subset, mask, 0.0)
        elif args.mask[i] == "bottom":
            mask = createMask(ds.metadata, "bottom")
            withMissing = SpecificFeatureRemovingDataset(subset, mask, 0.0)
        split_TransformedTraindatasets.append(withMissing)
    return split_TransformedTraindatasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=dict(), help='Parameters to load the dataset')
    parser.add_argument("--output", type=str, default="./models/nn/", help='Location to save final regressor')
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')
    parser.add_argument("--force_cpu", action='store_true',
                        help="If set, forces using the CPU for calculations instead of the GPU.")

    # training
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for Adam')
    parser.add_argument('--training_iterations', type=int, default=200,
                        help='Maximum training iterations, may train for less if the model stops improving')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during training')
    parser.add_argument('--validate_every', type=int, default=10,
                        help='How often to validate the model during training iteration')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of times validation can get worse before stopping training')
    parser.add_argument("--classification", action='store_true',
                        help="If set, runs in classification mode.")
    parser.add_argument("--evaluate_training", action='store_true',
                        help="If set, training accuracy is logged during validation. Useful for debugging patience.")

    # model parameters
    parser.add_argument("--input", type=str, default=None, help='Input model to continue training')
    parser.add_argument("--architecture", type=jsonOrString, default=None,
                        help='Neural network architecture to use')
    # TODO: ditch layers, its part of architecture now
    parser.add_argument("--layers", type=int, nargs='*', default=[],
                        help="Sizes of each linear layer in the model")
    # image processing
    parser.add_argument("--mask", type=json.loads, default = json.loads('["full", "top", "bottom"]'), help="Name of the mask to use")
    #WandB
    parser.add_argument('--wandb_username', type=str, default=None)


    args = parser.parse_args()

    ######  Weight and biases logging ############
    if args.wandb_username is not None:
        wandb.init(project="MissingValueTesting", entity=args.wandb_username)


    # start logging
    outputFolder = args.output
    date = setupLogging(args.verbose, os.path.join(outputFolder, "log"), args.name, args=args)
    logging.info(f"Starting to train {args.name}")

    # load in dataset
    ds = getDatasetSplits(args.name, **args.dataset)

    # seed random parameters
    torch.manual_seed(args.seed)  # TODO: anymore work for seeds?
    rand = Generator()
    rand.manual_seed(args.seed)

    # construct model
    logging.info("Constructing neural network")
    model: NeuralNetworkRegressor
    if args.input is not None:
        logging.info(f"Loading existing model from {args.input}")
        model = NeuralNetworkRegressor.load(args.input)
        # ensure the feature index is set, don't want to accidentally retrain with fewer features
        model.setFeatureIndex(-1)
    elif args.architecture is not None:
        model = createRegressorFromJson(ds, args.architecture)
    else:
        logging.warning("Using deprecated layers argument")
        model = createRegressorFromJson(ds, args.layers)

    logging.info(f"Network has {sum(p.numel() for p in model.nn.parameters() if p.requires_grad)} parameters")

    # other setup
    #lossFunction = BCEWithLogitsLoss() if args.classification else MSELoss()
    lossFunction = CrossEntropyLoss() if args.classification else MSELoss()
    optimizer = Adam(model.nn.parameters(), lr=args.learning_rate)

    # device setup
    device = torch.device("cuda" if not args.force_cpu and torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} for tensor calculations, cuda available: {torch.cuda.is_available()}")
    model.to(device)

    # setup data loading
    split_Traindatasets = split_dataset(ds.train, len(args.mask))
    split_TransformedTraindatasets = mask_split_dataset(split_Traindatasets,ds,args)

    combined_dataset = ConcatDataset(split_TransformedTraindatasets)
    ds.train = combined_dataset # Train dataset with a mix of corrupted and non-corrupted images #Better to have a local variable

    dataLoader = DataLoader(ds.train, batch_size=args.batch_size, shuffle=True, generator=rand, pin_memory=True)


    # TODO: different batch size?
    split_Valdatasets = split_dataset(ds.validate, len(args.mask))
    split_TransformedValdatasets = mask_split_dataset(split_Valdatasets, ds, args)

    if len(args.mask) > 1:
        combined_dataset = ConcatDataset(split_TransformedValdatasets)
        split_TransformedValdatasets.append(combined_dataset)

    validateLoader = [DataLoader(x, batch_size=args.batch_size, shuffle=False, generator=rand, pin_memory=True) for x in split_TransformedValdatasets] #Validate dataloader is a list

    testLoader = DataLoader(ds.test, batch_size=args.batch_size, shuffle=False, generator=rand)


    # evaluate the initial model
    model.nn.eval()
    if args.evaluate_training:
        trainingAccuracy = model.evaluateDataloader(dataLoader, device, lossFunction, args.seed)
        logging.info(f"Initial training error {trainingAccuracy.mean().item()}")

    validationError = model.evaluateDataloader(validateLoader[-1], device, lossFunction, args.seed)
    validationBest = validationError.mean().item()
    logging.info(f"Initial validation error {validationError.mean().item()}")

    # start training
    logging.info("Starting network learning")
    startTime = perf_counter()
    bestParams = copy.deepcopy(model.nn.state_dict())
    validationFails = 0

    numBatches = len(dataLoader)
    for i in range(args.training_iterations): # This is the total number of epochs
        iterationStart = perf_counter()
        model.nn.train()

        # standard training stuff
        totalLoss = 0
        for batchIndex, (features, targets) in enumerate(dataLoader):
            features = features.to(device)
            if lossFunction == CrossEntropyLoss():
                targets = process_tensor(targets, seed_local=args.seed)
            targets = targets.to(device)

            optimizer.zero_grad()

            prediction = model.predictWithGradient(features)
            loss: Tensor = lossFunction(prediction, targets)
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            print(f"Evaluating iteration {i + 1}/{args.training_iterations} for batch {batchIndex + 1}/{numBatches}",
                  end="\r")

        logging.info(f"{args.name} iteration {i + 1}/{args.training_iterations} in "
                     f"{perf_counter() - iterationStart:.5f} seconds - error: {totalLoss / numBatches}")

        model.nn.eval()
        trainAccuracy = model.evaluateAccDataloader(dataLoader, device, lossFunction)
        print(f"The trainAccuracy is: {trainAccuracy}")
        trainAccuracyNet = trainAccuracy.mean().item()

        trainingError = model.evaluateDataloader(dataLoader, device, lossFunction)
        trainingErrorNet = trainingError.mean().item()

        #wandb.log({"train_loss": trainingError, "val_loss": valError}, step=i)
        if args.wandb_username is not None:
            wandb.log({"train_loss": trainingError, "train_accuracy": trainAccuracyNet}, step=i)

        if args.wandb_username is not None:
            if i % 5 == 0:
                for index, y in enumerate(validateLoader):
                    valAccuracy = model.evaluateAccDataloader(y, device, lossFunction)
                    valAccuracyNet = valAccuracy.mean().item()

                    valError = model.evaluateDataloader(y, device, lossFunction)
                    valErrorNet = valError.mean().item()
                    if index == 0:
                        wandb.log({"val_full_accuracy": valAccuracyNet, "val_full_loss": valError}, step=i)
                    elif index == 1:
                        wandb.log({"val_top_accuracy": valAccuracyNet, "val_top_loss": valError}, step=i)
                    elif index == 2:
                        wandb.log({"val_bottom_accuracy": valAccuracyNet, "val_bottom_loss": valError}, step=i)
                    else:
                        wandb.log({"val_All_accuracy": valAccuracyNet, "val_All_loss": valError}, step=i)




        #wandb.log({"train_accuracy": trainAccuracyNet, "val_accuracy": valAccuracyNet}, step=i)

        # if this is the new best model, store it
        if i % args.validate_every == 0 or (i+1) == numBatches:
            logging.info(f"Evaluating the model via validation data")
            model.nn.eval()

            # save a copy of the model so far
            outputPath = os.path.join(outputFolder, f"{args.name}-{date}-{i+1}.pklz")
            logging.info(f"Saving model at iteration {i+1} to {outputPath}")
            model.save(outputPath)

            if args.evaluate_training:
                trainingAccuracy = model.evaluateDataloader(dataLoader, device, lossFunction)
                logging.info(f"Training error in evaluate mode {trainingAccuracy.mean().item()}")

            # FIXME: there is probably a better way to compare multiple variables
            validationError = model.evaluateDataloader(validateLoader[-1], device, lossFunction)
            validationErrorMean = validationError.mean().item()
            if validationErrorMean > validationBest:
                logging.info(f"Worsening on valid {validationErrorMean} > prev best {validationBest}")
                if validationFails >= args.patience:
                    logging.info(f"Exceeding patience {args.patience}, stopping training")
                    break
                else:
                    validationFails += 1
            else:
                logging.info(f'Found new best model with error {validationErrorMean} < prev best {validationBest}')
                bestParams = copy.deepcopy(model.nn.state_dict())
                validationBest = validationErrorMean
                validationFails = 0

    # restore best model
    endTime = perf_counter()
    logging.info(f"Network learning done in {endTime - startTime:.5f} secs")
    model.nn.load_state_dict(bestParams)

    # TODO: error history graph?
    """
    # save training performances
    perf_path = os.path.join(hist_dir, f'{str(percent * 100)}.train-hist')
    np.save(perf_path, train_history)
    logging.info(f'Training history saved to {perf_path}')

    #
    # and plot it
    perf_path = os.path.join(hist_dir, f'{str(percent * 100)}.train-hist.pdf')
    plt.plot(np.arange(len(train_history)), train_history)
    plt.savefig(perf_path)
    plt.close()
    """

    # save the model
    # start by resetting some properties, not sure if this is needed, but it feels safer before saving the model
    model.nn.eval()
    model.nn.cpu()
    model.nn.zero_grad()
    outputPath = os.path.join(outputFolder, f"{args.name}-{date}.pklz")
    logging.info(f"Saving model to {outputPath}")
    model.save(outputPath)

    # final evaluation of the model (done after saving as we don't need the result to save, and it might be slow)
    model.nn.to(device)
    model.evaluateDataLoaders(dataLoader, validateLoader, testLoader, device, lossFunction, "BCE" if args.classification else "MSE")

    if args.wandb_username is not None:
        wandb.finish()
