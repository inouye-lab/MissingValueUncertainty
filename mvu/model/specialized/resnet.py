import copy
import logging

import torch
from torch import nn, Tensor
from torch.nn import Module, Conv2d
from torchvision.models import resnet18, ResNet18_Weights


class ExponentialActivation(Module):
    """Activation function using simple exponential."""
    # noinspection PyMethodMayBeStatic
    def forward(self, tensor: Tensor) -> Tensor:
        return torch.exp(tensor)


class Resnet18Classifier(Module):
    resnet: resnet18
    """Resnet module instance"""
    numClasses: int
    """Number of classes for the output"""

    activation: nn.Module
    """Final activation function"""

    def __init__(self, num_classes: int, momentum: float = None, track_running_stats: bool = None,
                 pretrained_weights: bool = True, activation: str = "sigmoid", keep_final_layer: bool = False,
                 num_channels: int = 3, copy_input_weights: bool = True,
                 resnet: resnet18 = None):
        """
        Creates a new instance of the classifier
        :param num_classes:            Number of output classes to use
        :param momentum:              If set, overrides the momentum property in all BatchNorm2d layers
        :param track_running_stats:   If set, overrides the track_running_stats property in all BatchNorm2d layers
        :param pretrained_weights:    If true, uses `ResNet18_Weights.DEFAULT` for the starting weights. False uses none
        :param keep_final_layer:      If true, keeps the weights of the final layer. Class count must match to use
        :param resnet:                Existing resnet architecture to copy. If None, creates one
        """
        super().__init__()

        # repurposing resnet for the classifier
        # swap out the final layer for our new class list
        if resnet is not None:
            logging.info("Using passed Resnet model")
            self.resnet = resnet
        else:
            logging.info(f"Creating new Resnet with {'pretrained' if pretrained_weights else 'randomized'} weights")
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_weights else None)
        if momentum is not None or track_running_stats is not None:
            for module in self.resnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if momentum is not None:
                        module.momentum = momentum
                    if track_running_stats is not None:
                        module.track_running_stats = track_running_stats
        self.numClasses = num_classes
        if keep_final_layer:
            assert self.resnet.fc.out_features == self.numClasses,\
                f"Cannot keep final layer if the class count differs; layer size {self.resnet.fc.out_features} for class count {self.numClasses}"
            logging.info("Keeping original final layer for Resnet model")
        else:
            logging.info(f"Replacing final layer in Resnet for {num_classes} classes")
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        if num_channels != 3 or not copy_input_weights:
            self.resnet.conv1 = _createResnetConv2dWithChannels(self.resnet.conv1, num_channels, copyWeights=copy_input_weights)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "identity":
            self.activation = nn.Identity()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "exp":
            self.activation = ExponentialActivation()
        else:
            raise ValueError(f"Unknown activation function '{activation}'")

    def forward(self, features: Tensor):
        # step 1: apply model
        features = self.resnet(features)
        # step 2: apply activation
        features = self.activation(features)
        return features


def _createResnetConv2dWithChannels(original: Conv2d, channels: int, copyWeights: bool = False):
    layer = Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if copyWeights:
        logging.info("Copying original first layer weights for Resnet DMV")
        # if we have fewer than 3 target channels, then copy just the number that fit
        if channels < 3:
            layer.weight.data[:,0:channels,:,:] = original.weight.data[:,0:channels,:,:].clone()
        else:
            # with 3 or more, copy the full list and 0 out anything remaining
            layer.weight.data[:,0:3,:,:] = original.weight.data.clone()
            if channels > 3:
                layer.weight.data[:,3:channels,:,:] = 0
    else:
        logging.info("Using randomized first layer weights for Resnet DMV")
    return layer


def _flattenSingleClass(num_classes: int) -> int:
    """Converts a single class into two classes as needed"""
    if num_classes == 1:
        return 2
    return num_classes


class Resnet18Dirichlet(Resnet18Classifier):
    """
    Resnet structure that inputs a 4 channel image (with missingness) and outputs a strength vector
    """
    def __init__(self, num_classes: int, minStrength: float = 1e-35, activation: str = "softplus", copy_input_weights: bool = False, num_channels: int = 3, *args, **kwargs):
        # add 1 channel for the missing layer to the input space. Default to not copying input weights
        super().__init__(_flattenSingleClass(num_classes), *args, copy_input_weights=copy_input_weights, num_channels=num_channels+1, activation=activation, **kwargs)
        self.minStrength = minStrength

    def forward(self, features: Tensor):
        # step 1: apply model
        strengths = self.resnet(features)
        # step 2: send strengths through activation. Clamp is just here for safety in case we go to small
        strengths = torch.clamp(self.activation(strengths), min=self.minStrength)
        # return the strengths alone
        return strengths

    @classmethod
    def fromResnet(cls, classifier: Resnet18Classifier, *args, keep_final_layer: bool = True, copy_input_weights: bool = True, **kwargs):
        """Copies a resnet classifier to create a new dirichlet network"""
        # by default, we want to keep final and input layer as the goal of copying is as similar of a model as possible
        return cls(classifier.numClasses, *args, resnet=copy.deepcopy(classifier.resnet),
                   keep_final_layer=keep_final_layer, copy_input_weights=copy_input_weights, **kwargs)


class Resnet18DirichletStrength(Resnet18Classifier):
    """
    Resnet structure that inputs a 4 channel image (with missingness) and outputs a probability vector plus strength.
    """
    def __init__(self, num_classes: int, minStrength: float = 1e-35, copy_input_weights: bool = False, *args, **kwargs):
        self.numClasses = _flattenSingleClass(num_classes)
        # added an extra 1 class for the final fully connected layer
        super().__init__(self.numClasses + 1, *args, **kwargs)
        # swap out first layer for one with 4 channels, 4th is missing
        self.resnet.conv1 = _createResnetConv2dWithMissing(self.resnet.conv1, copyWeights=copy_input_weights)
        self.minStrength = minStrength

    def forward(self, features: Tensor):
        # step 1: apply model
        features = self.resnet(features)
        # step 2: send num_classes features through the standard activation
        probabilities = self.activation(features[:, 0:self.numClasses])
        # step 3: send strength through a standard relu
        strength = torch.clamp(features[:, self.numClasses], min=self.minStrength)

        # return the pair, lets the operator decide to multiply them or keep them separate
        return probabilities, strength

