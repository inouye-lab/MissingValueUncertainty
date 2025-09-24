import argparse
import copy
import json
import logging
import os
from time import perf_counter
from typing import List, Optional

import torch
from torch import Tensor, Generator, nn
from torch.nn import Identity, BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset

from mvu.dataset.loader import getDatasetSplits
from mvu.dataset.mutators import IncludeMask, randomDropping, MaskedDataset
from mvu.dataset.mutators import createMask, RandomMaskedDataset, distributeMasks
from mvu.logger import setupLogging
from mvu.model.loader import createRegressorFromJson
from mvu.model.loss import DirichletLoss, DirichletStrengthLogitLoss, DirichletLogitLoss
from mvu.model.regressor import NeuralNetworkRegressor, Regressor
from mvu.model.specialized.resnet import Resnet18DirichletStrength
from mvu.util import jsonOrString, selectDevice, jsonOrName, getOptimizer, getScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("name", type=str, help='Name of the dataset to parse')
    parser.add_argument("dataset", type=json.loads, default=dict(), help='Parameters to load the dataset')
    parser.add_argument("--output", type=str, default="./models/nn/", help='Location to save final regressor')
    parser.add_argument('--seed', type=int, default=1337, help='Seed for random permutations')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1, help='Logging verbosity level')
    parser.add_argument("--cuda_index", type=int, default=0,
                        help="Index to use for CUDA, set to -1 to force CPU")

    # missing features
    parser.add_argument("--masks", type=jsonOrName, nargs="*", default=[], help="Name of the masks to use")
    parser.add_argument("--drop", type=jsonOrName, default=dict(), help="Parameters for dropping for the dataset")

    # training
    parser.add_argument('--training_iterations', type=int, default=200,
                        help='Maximum training iterations, may train for less if the model stops improving')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during training')
    parser.add_argument('--validate_every', type=int, default=10,
                        help='How often to validate the model during training iteration')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of times validation can get worse before stopping training')
    parser.add_argument("--evaluate_training", action='store_true',
                        help="If set, training accuracy is logged during validation. Useful for debugging patience.")
    parser.add_argument("--teacher", type=str, default=None, help='Path to the pretrained regressor to load')

    # optimizer
    parser.add_argument("--optimizer", type=jsonOrName, default="adam", help='Optimizer choice')
    parser.add_argument("--scheduler", type=jsonOrName, default=dict(), help='Scheduler configuration')

    # loss function config
    parser.add_argument('--masked_weight', type=float, default=0.5,
                        help='Weight for the masked loss term')
    parser.add_argument('--dirichlet_weight', type=float, default=0.5,
                        help='Weight for the dirichlet loss term')
    parser.add_argument('--clean_weight', type=float, default=1,
                        help='Weight for the clean loss term')
    parser.add_argument('--use_logits', action='store_true',
                        help='If true, uses the logit variant of the loss function. Expected to run the model with no activation function')

    # model parameters
    parser.add_argument("--input", type=str, default=None, help='Input model to continue training')
    parser.add_argument("--architecture", type=jsonOrString, default=None,
                        help='Neural network architecture to use')

    args = parser.parse_args()

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

    # device setup
    device = selectDevice(args.cuda_index)

    # load in the teacher
    teacher: Optional[NeuralNetworkRegressor] = None
    if args.teacher is not None:
        logging.info(f"Using teacher model from {args.teacher}")
        teacher = NeuralNetworkRegressor.load(args.teacher)
        teacher.to(device)
        teacher.nn.eval()
        # TODO: is there a way to not hardcode this?
        if not args.use_logits:
            if len(ds.metadata.labels) == 1:
                teacher.activation = nn.Sigmoid()
            else:
                teacher.activation = nn.Softmax(dim=1)

    # construct model
    logging.info("Constructing neural network")
    model: NeuralNetworkRegressor
    if args.input is not None:
        logging.info(f"Loading existing model from {args.input}")
        model = NeuralNetworkRegressor.load(args.input)
        # ensure the feature index is set, don't want to accidentally retrain with fewer features
        model.setFeatureIndex(-1)
    elif args.architecture is not None:
        model = createRegressorFromJson(ds, args.architecture, original=teacher)
    else:
        logging.error("Must set either input or architecture")
        exit(1)
    model.to(device)

    # ensure the teacher is set
    if teacher is None:
        logging.error("Using new model to predict phi values")
        teacher = model


    logging.info(f"Network has {sum(p.numel() for p in model.nn.parameters() if p.requires_grad)} parameters")

    # different loss function based on model architecture
    if isinstance(model.nn, Resnet18DirichletStrength):
        lossFunction = DirichletStrengthLogitLoss(args.masked_weight, args.dirichlet_weight, cleanWeight=args.clean_weight)
        model.nn.activation = Identity()
    elif args.use_logits:
        cleanLoss = BCEWithLogitsLoss() if len(ds.metadata.target) == 1 else CrossEntropyLoss()
        lossFunction = DirichletLogitLoss(args.masked_weight, args.dirichlet_weight, cleanWeight=args.clean_weight, cleanLoss=cleanLoss)
    else:
        lossFunction = DirichletLoss(args.masked_weight, args.dirichlet_weight, cleanWeight=args.clean_weight)

    # other setup
    optimizer = getOptimizer(model.nn, **args.optimizer)
    scheduler = getScheduler(optimizer, **args.scheduler)

    # setup mutator
    maskedTraining: Dataset
    maskedValidation: Dataset
    # when we have a teacher, don't give the teacher data with the mask
    includeMask = IncludeMask.ALWAYS if args.teacher is None else IncludeMask.MISSING

    commonMaskArgs = dict(missingValue=0, includeMask=includeMask, returnOriginal=True)

    # clean validation data for accuracy check
    cleanValidation = MaskedDataset(ds.validate, createMask(ds.metadata, "none"), missingValue=0, includeMask=includeMask)

    masks: List[Tensor] = []
    if len(args.masks) > 0:
        logging.info(f"Using masked missingness with {args.masks}")
        masks = [createMask(ds.metadata, **mask) for mask in args.masks]
        # for training, randomly choose mask
        maskedTraining = RandomMaskedDataset(ds.train, masks, rand, **commonMaskArgs)
        # for validation, split the set into parts using each mask
        maskedValidation = distributeMasks(ds.validate, masks, rand, **commonMaskArgs)
    else:
        logging.info(f"Using random dropping with arguments {args.drop}")
        maskedTraining   = randomDropping(ds.train,    ds.metadata, **commonMaskArgs, **args.drop)
        maskedValidation = randomDropping(ds.validate, ds.metadata, **commonMaskArgs, **args.drop)

    # setup data loading
    trainLoader    = DataLoader(maskedTraining,   batch_size=args.batch_size, shuffle=True,  generator=rand, pin_memory=True)
    validateLoader = DataLoader(maskedValidation, batch_size=args.batch_size, shuffle=False, generator=rand, pin_memory=True)
    cleanLoader    = DataLoader(cleanValidation,  batch_size=args.batch_size, shuffle=False, generator=rand, pin_memory=True)

    # logic to handle evaluating batches
    def batchHandler(maskedFeatures: Tensor, cleanFeatures: Tensor, targets: Tensor):
        maskedFeatures = maskedFeatures.to(device)
        cleanFeatures = cleanFeatures.to(device)
        targets = targets.to(device)
        maskedPrediction = model.predict(maskedFeatures)
        cleanPrediction = teacher.predict(cleanFeatures)
        # might at that point want multiple loss function support
        loss = lossFunction(cleanPrediction, maskedPrediction, targets).item()
        return loss, targets.shape[0]

    if len(ds.metadata.target) == 1:
        if args.use_logits:
            def accuracyLoss(prediction: Tensor, targets: Tensor) -> Tensor:
                return torch.eq(sigmoid(prediction) > 0.5, targets).float().mean(dim=0)
        else:
            def accuracyLoss(prediction: Tensor, targets: Tensor) -> Tensor:
                return torch.eq(prediction > 0.5, targets).float().mean(dim=0)
    else:
        def accuracyLoss(prediction: Tensor, targets: Tensor) -> Tensor:
            return torch.eq(prediction.max(dim=1).indices, targets).float().mean(dim=0)

    # evaluate the initial model
    model.nn.eval()
    if args.evaluate_training:
        trainingAccuracy = Regressor.evaluateData(trainLoader, batchHandler)
        logging.info(f"Initial training error {trainingAccuracy.mean().item()}")

    validationAccuracy = model.evaluateDataloader(cleanLoader, device=device, lossFunction=accuracyLoss)
    logging.info(f"Initial validation accuracy {validationAccuracy}")

    validationError = Regressor.evaluateData(validateLoader, batchHandler)
    validationBest = validationError.mean().item()
    logging.info(f"Initial validation error {validationBest}")

    # start training
    logging.info("Starting network learning")
    startTime = perf_counter()
    bestParams = copy.deepcopy(model.nn.state_dict())
    validationFails = 0

    numBatches = len(trainLoader)
    for i in range(args.training_iterations):
        iterationStart = perf_counter()
        model.nn.train()

        # standard training stuff
        totalLoss = 0
        for batchIndex, (maskedFeatures, cleanFeatures, targets) in enumerate(trainLoader):
            maskedFeatures = maskedFeatures.to(device)
            cleanFeatures = cleanFeatures.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            maskedPrediction = model.predictWithGradient(maskedFeatures)
            if args.clean_weight > 0 or args.dirichlet_weight > 0:
                if args.teacher is None:
                    cleanPrediction = model.predictWithGradient(cleanFeatures)
                else:
                    cleanPrediction = teacher.predict(cleanFeatures)
            else:
                # if we aren't going to use clean, save some effort by just reusing the masked prediction
                cleanPrediction = torch.ones_like(maskedPrediction)
            loss: Tensor = lossFunction(cleanPrediction, maskedPrediction, targets)
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
            print(f"Evaluating iteration {i + 1}/{args.training_iterations} for batch {batchIndex + 1}/{numBatches}",
                  end="\r")

        logging.info(f"{args.name} iteration {i + 1}/{args.training_iterations} in "
                     f"{perf_counter() - iterationStart:.5f} seconds - error: {totalLoss / numBatches}")

        # if this is the new best model, store it
        if i % args.validate_every == 0 or (i+1) == numBatches:
            logging.info(f"Evaluating the model via validation data")
            model.nn.eval()

            # save a copy of the model so far
            outputPath = os.path.join(outputFolder, f"{args.name}-{date}-{i+1}.pklz")
            logging.info(f"Saving model at iteration {i+1} to {outputPath}")
            model.save(outputPath)

            if args.evaluate_training:
                trainingAccuracy = Regressor.evaluateData(trainLoader, batchHandler)
                logging.info(f"Training error in evaluate mode {trainingAccuracy.mean().item()}")
            validationAccuracy = model.evaluateDataloader(cleanLoader, device=device, lossFunction=accuracyLoss)
            logging.info(f"Validation accuracy: {validationAccuracy}")

            # FIXME: there is probably a better way to compare multiple variables
            validationError = Regressor.evaluateData(validateLoader, batchHandler)
            validationErrorMean = validationError.mean().item()
            if validationErrorMean > validationBest:
                logging.info(f"Worsening on valid {validationErrorMean} > prev best {validationBest}, patience {validationFails} / {args.patience}")
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

        # increment the scheduler
        if scheduler is not None:
            scheduler.step()

    # restore best model
    endTime = perf_counter()
    logging.info(f"Network learning done in {endTime - startTime:.5f} secs")
    model.nn.load_state_dict(bestParams)

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
    if len(masks) > 0:
        maskedTest = distributeMasks(ds.test, masks, rand, **commonMaskArgs)
    else:
        maskedTest = randomDropping(ds.test, ds.metadata, **commonMaskArgs, **args.drop)

    testLoader = DataLoader(maskedTest, batch_size=args.batch_size, shuffle=False, generator=rand, pin_memory=True)
    for (name, loader) in [("train", trainLoader), ("validate", validateLoader), ("test", testLoader)]:
        result = Regressor.evaluateData(loader, batchHandler)
        logging.info(f"Loss for {name} is {result.mean().item()}:\n{result}")
