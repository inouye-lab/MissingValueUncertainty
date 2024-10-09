from torch import nn, Tensor
from torch.nn import Module
from torchvision.models import resnet18, ResNet18_Weights


class Resnet18Classifier(Module):
    resnet: resnet18
    """Resnet module instance"""
    numClasses: int
    """Number of classes for the output"""

    activation: nn.Module
    """Final activation function"""

    def __init__(self, num_classes: int, momentum: float = None, track_running_stats: bool = None,
                 pretrained_weights: bool = True):
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
        self.activation = nn.Sigmoid()

    def forward(self, features: Tensor):
        # step 1: apply model
        features = self.resnet(features)
        # step 2: apply activation
        #features = self.activation(features) #Deactivating to apply BCEwithLogitsLoss
        return features

