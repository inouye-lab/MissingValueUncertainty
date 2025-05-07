import torch
from torch import nn, Tensor
from torch.nn import Module, Conv2d
from torchvision.models import resnet18, ResNet18_Weights


class Resnet18Classifier(Module):
    resnet: resnet18
    """Resnet module instance"""
    numClasses: int
    """Number of classes for the output"""

    activation: nn.Module
    """Final activation function"""

    def __init__(self, num_classes: int, momentum: float = None, track_running_stats: bool = None,
                 pretrained_weights: bool = True, activation: str = "sigmoid"):
        """
        Creates a new instance of the classifier
        :param num_classes:            Number of output classes to use
        :param momentum:              If set, overrides the momentum property in all BatchNorm2d layers
        :param track_running_stats:   If set, overrides the track_running_stats property in all BatchNorm2d layers
        :param pretrained_weights:    If true, uses `ResNet18_Weights.DEFAULT` for the starting weights. False uses none
        """
        super().__init__()

        # repurposing resnet for the classifier
        # swap out the final layer for our new class list
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_weights else None)
        if momentum is not None or track_running_stats is not None:
            for module in self.resnet.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if momentum is not None:
                        module.momentum = momentum
                    if track_running_stats is not None:
                        module.track_running_stats = track_running_stats
        self.numClasses = num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "identity":
            self.activation = nn.Identity()

    def forward(self, features: Tensor):
        # step 1: apply model
        features = self.resnet(features)
        # step 2: apply activation
        features = self.activation(features)
        return features


def _createResnetConv2dWithMissing():
    return Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


def _flattenSingleClass(num_classes: int) -> int:
    """Converts a single class into two classes as needed"""
    if num_classes == 1:
        return 2
    return num_classes


class Resnet18Dirichlet(Resnet18Classifier):
    """
    Resnet structure that inputs a 4 channel image (with missingness) and outputs a strength vector
    """
    def __init__(self, num_classes: int, minStrength: float = 1e-10, *args, **kwargs):
        super().__init__(_flattenSingleClass(num_classes), *args, **kwargs)
        # swap out first layer for one with 4 channels, 4th is missing
        self.resnet.conv1 = _createResnetConv2dWithMissing()
        self.minStrength = minStrength

    def forward(self, features: Tensor):
        # step 1: apply model
        strengths = self.resnet(features)
        # step 2: send strengths through a clamp
        strengths = torch.exp(strengths)
        # return the strengths alone
        return strengths


class Resnet18DirichletStrength(Resnet18Classifier):
    """
    Resnet structure that inputs a 4 channel image (with missingness) and outputs a probability vector plus strength.
    """
    def __init__(self, num_classes: int, minStrength: float = 1e-10, *args, **kwargs):
        self.numClasses = _flattenSingleClass(num_classes)
        # added an extra 1 class for the final fully connected layer
        super().__init__(self.numClasses + 1, *args, **kwargs)
        # swap out first layer for one with 4 channels, 4th is missing
        self.resnet.conv1 = _createResnetConv2dWithMissing()
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

