import logging
from typing import Dict, Union, Optional

from mvu.dataset.csv import CsvDatasetSplits
from mvu.dataset.specialized.starcraft import createStarCraftDataset
from mvu.dataset.torch import TorchDatasetSplits


def getDatasetSplits(name: str, path: str = None, **kwargs) -> TorchDatasetSplits:
    """
    Base method for loading datasets
    :param name:     Name of the dataset to load, has special behavior for "starcraft"
    :param path:     Path to the dataset, if unset can infer from name
    :param kwargs:   Additional arguments for starcraft dataset.
    :return:  Dataset splits for experiments
    """
    # custom loading logic for certain datasets
    if name == "starcraft":
        logging.info(f"Loading StarCraft dataset from {path}")
        return createStarCraftDataset(path=path, **kwargs)

    # determine the path from the arguments
    if path is None:
        path = f"./datasets/binary/{name}.pklz"
    logging.info(f"Loading CSV dataset from {path}")
    return CsvDatasetSplits.load(path).toTorch()
