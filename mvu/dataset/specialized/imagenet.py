import itertools
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Dict

import scipy
from torchvision import transforms
from torchvision.datasets import ImageNet

from ..meta import ImageDatasetMeta
from ..torch_utils import TorchDatasetSplits


def createImageNetDataset(path: str = None, validation_percent: float = 0.1, samples: Dict[str,int] = None, sensor_size: int = 1) -> TorchDatasetSplits:
    """
    Helper to load in ImageNet in the format we expect.
    :param path:                Location to load the imagenet dataset into.
    :param validation_percent:  Percentage of training data to use for validation.
    :param samples:             Maximum samples from the dataset to use for train, validate, and test.
    :param sensor_size:         Size of sensors for making values missing.
    :param targets:             Number of targets to use for training.
    :return:  Dataset instance
    """
    assert path is not None, "Must pass in a path to use the ImageNet dataset"
    # insure the path exists, just to be safe
    Path(path).mkdir(exist_ok=True, parents=True)

    # locate class names
    metaPath = os.path.join(path, "ILSVRC2012_devkit_t12.tar.gz")
    targets: List[str]
    with tarfile.open(metaPath, "r:gz") as tar:
        with tar.extractfile("ILSVRC2012_devkit_t12/data/meta.mat") as meta:
            # Start by using loadmat, which supports filelike
            # noinspection PyTypeChecker
            mat = scipy.io.loadmat(meta)
            # iterate over the synsets in the meta
            synsets = mat["synsets"]
            # create list of classes
            targets = list(itertools.repeat("", 1000))
            for i in range(1000):
                classData = synsets[i,0]
                assert classData['ILSVRC2012_ID'].item() == i + 1, f"Found wrong class index in synsets: expected {i + 1} but got {classData['ILSVRC2012_ID']}"
                assert classData['num_train_images'].item() > 0, f"Found 0 training images for class {i + 1}"
                targets[i] = classData['words'].item()
            for i in range(1000, synsets.shape[0]):
                assert synsets[i,0]['num_train_images'].item() == 0, f"Found training images for class {i + 1} larger than 1000"

    # we just know the classes directly, not sure if this can be fetched
    logging.info(f"Using {len(targets)} ImageNet targets, first 20: {targets[0:20]}")

    # create metadata
    meta = ImageDatasetMeta("imagenet", targets, 224, sensor_size, 3)

    # fetch ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainingValidation = ImageNet(path, "train", transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))
    testing = ImageNet(path, "val", transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]))

    return TorchDatasetSplits.split(meta, trainingValidation, testing, validation_percent=validation_percent, samples=samples)
