import json
import logging
from typing import List, Union, Dict

from torch.nn import Linear, ReLU, Flatten, Sequential, Module

from .diffusion import GaussianDiffusionBatchGenerator
from .generator import BatchGenerator
from .regressor import NeuralNetworkRegressor
from .specialized.resnet import Resnet18Classifier, Resnet18Dirichlet, Resnet18DirichletStrength
from .specialized.image import ImageRegressor
from ..dataset.meta import ImageDatasetMeta
from ..dataset.torch_utils import TorchDatasetSplits

def _ensureNumClassesSet(name: str, ds: TorchDatasetSplits, kwargs: Dict):
        # if not set, use the dataset for info on number of classes
    if "num_classes" not in kwargs:
        if isinstance(ds.metadata.target, list):
            kwargs["num_classes"] = len(ds.metadata.target)
        else:
            kwargs["num_classes"] = 1
    logging.info(f"Constructing {name} with {kwargs['num_classes']} targets")

def createRegressor(ds: TorchDatasetSplits, name, original: NeuralNetworkRegressor = None, **kwargs) -> NeuralNetworkRegressor:
    """
    Creates a new neural network model
    :param ds:       Dataset
    :param name:     Architecture name
    :param kwargs:   Architecture arguments
    :param original: Existing model to potentially copy
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
        logging.info(f"Using Resnet18 architecture with {kwargs}")
        _ensureNumClassesSet("ResNet", ds, kwargs)
        return NeuralNetworkRegressor(Resnet18Classifier(**kwargs))
    elif name == "resnet_dirichlet":
        logging.info(f"Using Resnet18 DMV architecture with {kwargs}")
        _ensureNumClassesSet("ResNet Dirichlet", ds, kwargs)
        return NeuralNetworkRegressor(Resnet18Dirichlet(**kwargs))
    elif name == "copy_resnet_dirichlet":
        logging.info(f"Copying to create Resnet18 DMV architecture with {kwargs}")
        assert original is not None, "Cannot copy if no original model passed"
        assert isinstance(original.nn, Resnet18Classifier), "Copy resnet requires a resnet classifier to start"
        return NeuralNetworkRegressor(Resnet18Dirichlet.fromResnet(original.nn, **kwargs))
    elif name == "resnet_dirichlet_strength":
        _ensureNumClassesSet("ResNet Dirichlet Strength", ds, kwargs)
        return NeuralNetworkRegressor(Resnet18DirichletStrength(**kwargs))
    elif name == "simple_fully_connected":
        lastSize = ds.metadata.numInputs
        layers = kwargs["layers"]
        logging.info(f"Constructing model with input size {lastSize} and hidden layers {layers}")
        components: List[Module] = []
        for layer in layers:
            components.append(Linear(lastSize, layer))
            components.append(ReLU())
            lastSize = layer
        components.append(Linear(lastSize, 1))
        components.append(Flatten(start_dim=0))
        return NeuralNetworkRegressor(Sequential(*components))
    else:
        raise ValueError(f"Unknown neural network architecture '{name}'")


def createRegressorFromJson(ds: TorchDatasetSplits, value: Union[str, List, Dict], original: NeuralNetworkRegressor = None) -> NeuralNetworkRegressor:
    """
    Creates a new model using the passed JSON data
    :param ds:      Dataset
    :param value:   Value parsed from JSON argument
    :param original: Existing model to potentially copy
    :return: Model instance
    """
    if isinstance(value, dict):
        return createRegressor(ds, original=original, **value)
    if isinstance(value, list):
        return createRegressor(ds, name="simple_fully_connected", layers=value, original=original)
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
