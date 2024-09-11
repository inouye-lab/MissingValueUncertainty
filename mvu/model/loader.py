import json
import logging
from typing import List, Union, Dict

from torch.nn import Linear, ReLU, Flatten, Sequential, Module

from .diffusion import GaussianDiffusionBatchGenerator
from .generator import BatchGenerator
from .regressor import NeuralNetworkRegressor
from .specialized.resnet import Resnet18Classifier
from .specialized.image import ImageRegressor
from ..dataset.meta import ImageDatasetMeta
from ..dataset.torch import TorchDatasetSplits


def createRegressor(ds: TorchDatasetSplits, name, **kwargs) -> NeuralNetworkRegressor:
    """
    Creates a new neural network model
    :param ds:       Dataset
    :param name:     Architecture name
    :param kwargs:   Architecture arguments
    :return: Model instance
    """
    if name == "image_regression":
        if isinstance(ds.metadata, ImageDatasetMeta):
            logging.info(f"Using Image Regressor architecture with {ds.metadata.channels} channels, "
                         f"{ds.metadata.imageSize} image size and {len(ds.metadata.target)} outputs")
            return NeuralNetworkRegressor(
                ImageRegressor(ds.metadata.channels, ds.metadata.imageSize, len(ds.metadata.target))
            )
        else:
            raise ValueError("image-regression requires the dataset metadata to also be image metadata")
    elif name == "resnet":
        # if not set, use the dataset for info on number of classes
        if "num_classes" not in kwargs:
            if isinstance(ds.metadata.target, list):
                kwargs["num_classes"] = len(ds.metadata.target)
            else:
                kwargs["num_classes"] = 1
        logging.info(f"Constructing ResNet with {kwargs['num_classes']} targets")
        return NeuralNetworkRegressor(Resnet18Classifier(**kwargs))
    elif name == "simple_fully_connected":
        lastSize = ds.metadata.numInputs
        logging.info(f"Constructing model with input size {lastSize} and hidden layers {args.layers}")
        components: List[Module] = []
        for layer in kwargs["layers"]:
            components.append(Linear(lastSize, layer))
            components.append(ReLU())
            lastSize = layer
        components.append(Linear(lastSize, 1))
        components.append(Flatten(start_dim=0))
        return NeuralNetworkRegressor(Sequential(*components))
    else:
        raise ValueError(f"Unknown neural network architecture '{name}'")


def createRegressorFromJson(ds: TorchDatasetSplits, value: Union[str, List, Dict]) -> NeuralNetworkRegressor:
    """
    Creates a new model using the passed JSON data
    :param ds:      Dataset
    :param value:   Value parsed from JSON argument
    :return: Model instance
    """
    if isinstance(value, dict):
        return createRegressor(ds, **value)
    if isinstance(value, list):
        return createRegressor(ds, name="simple_fully_connected", layers=value)
    return createRegressor(ds, name=value)


def createBatchGenerator(name: str, **kwargs) -> BatchGenerator:
    """
    Creates a new batch generator instance using the given method
    :param name:     Generator name
    :param kwargs:   Other generator arguments
    :return:  generator instance
    """
    if name == "gaussian-diffusion":
        return GaussianDiffusionBatchGenerator(**kwargs)

    raise ValueError(f"Unknown generator type '{name}'")
