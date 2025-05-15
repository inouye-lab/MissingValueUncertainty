import logging

from mvu.dataset.specialized.baselines import createCIFAR10Dataset
from .csv import CsvDatasetSplits
from .specialized.celeba import createCelebADataset
from .specialized.starcraft import createStarCraftDataset
from .torch_utils import TorchDatasetSplits


def getDatasetSplits(name: str, path: str = None, **kwargs) -> TorchDatasetSplits:
    """
    Base method for loading datasets
    :param name:     Name of the dataset to load, has special behavior for "starcraft" and "celeba"
    :param path:     Path to the dataset, if unset can infer from name
    :param kwargs:   Additional arguments for the dataset.
    :return:  Dataset splits for experiments
    """
    # custom loading logic for certain datasets
    if name == "cifar10":
        logging.info(f"Loading CIFAR10 dataset from {path}")
        return createCIFAR10Dataset(path=path, **kwargs)
    if name == "starcraft":
        logging.info(f"Loading StarCraft dataset from {path}")
        return createStarCraftDataset(path=path, **kwargs)
    if name == "celeba":
        logging.info(f"Loading CelebA dataset from {path}")
        return createCelebADataset(images_root=path, **kwargs)

    # determine the path from the arguments
    if path is None:
        path = f"./datasets/binary/{name}.pklz"
    logging.info(f"Loading CSV dataset from {path}")
    return CsvDatasetSplits.load(path).toTorch()
