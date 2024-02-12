from torch import nn, Tensor
from torch.nn import Module


class ImageRegressor(Module):
    """Regression neural network designed for the starcraft image set"""

    inputChannels: int
    """Number of channels for the input image"""
    imageSize: int
    """Number of pixels forming the width and height of the image"""

    convolution: nn.Module
    """Convolution layers, maps from [samples]x3x64x64 to [samples]x128x15x15"""

    fullyConnectedSize: int
    """Output size from the convolution and input to the fully connected layer"""

    fullyConnected: nn.Module
    """Fully connected """

    def __init__(self, channels: int, imageSize: int, outputSize: int):
        super().__init__()
        # step 1: reshape back into a tensor
        self.inputChannels = channels
        self.imageSize = imageSize
        # step 2: apply convolution
        self.convolution = nn.Sequential(
            # size is [samples]x3x64x64
            nn.Conv2d(in_channels=self.inputChannels, out_channels=128, kernel_size=3, padding=1),
            # size is [samples]x128x64x64
            nn.MaxPool2d(kernel_size=(2, 2)),
            # size is [samples]x128x32x32
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            # size is [samples]x128x30x30
            nn.MaxPool2d(kernel_size=(2, 2)),
            # size is [samples]x128x15x15
        )
        # step 3: reshape into a vector
        imagePixels = int((imageSize / 4) - 1) ** 2
        self.fullyConnectedSize = 128*imagePixels
        # step 4: apply fully connected layers
        self.fullyConnected = nn.Sequential(
            nn.Linear(self.fullyConnectedSize, imagePixels),
            nn.ReLU(),
            nn.Linear(imagePixels, outputSize)
        )

    def forward(self, features: Tensor):
        sampleCount = features.shape[0]
        # step 1: reshape back into a tensor
        features = features.reshape(sampleCount, self.inputChannels, self.imageSize, self.imageSize)
        # step 2: apply convolution
        features = self.convolution(features)
        # step 3: reshape into a vector
        features = features.reshape(sampleCount, self.fullyConnectedSize)
        # step 4: apply fully connected
        features = self.fullyConnected(features)
        return features
