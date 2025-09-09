import math

from torch import Tensor
from torch.utils.data import Dataset

from mvu.dataset.mutators import DatasetWrapper, T_co


def split_batch_into_patches(image: Tensor, n_device: int) -> Tensor:
    B, C, H, W = image.shape
    n_patch_per_dim = int(math.sqrt(n_device))
    assert H % n_patch_per_dim == 0, 'd_img should be divisible by n_device'
    patch_size = H // n_patch_per_dim

    # save for later
    #image = image[:, :n_patch_per_dim * patch_size, :n_patch_per_dim * patch_size]
    image = image.view(B, C, n_patch_per_dim, patch_size, n_patch_per_dim, patch_size)
    image = image.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, n_device, C, patch_size, patch_size)

    return image

def split_image_into_patches(image: Tensor, n_device: int) -> Tensor:
    C, H, W = image.shape
    n_patch_per_dim = int(math.sqrt(n_device))
    assert H % n_patch_per_dim == 0, 'd_img should be divisible by n_device'
    assert H == W, "Image must be square"
    patch_size = H // n_patch_per_dim

    # save for later
    image = image.view(C, n_patch_per_dim, patch_size, n_patch_per_dim, patch_size)
    image = image.permute(1, 3, 0, 2, 4).contiguous().view(n_device, C, patch_size, patch_size)

    return image


class DeviceDataset(DatasetWrapper[T_co]):
    devices: int
    """Number of devices, should be a square number"""

    def __init__(self, base: Dataset[T_co], devices: int):
        super().__init__(base)
        self.devices = devices

    def __getitem__(self, item):
        data = self.base[item]
        # noinspection PyRedundantParentheses
        return (split_image_into_patches(data[0], self.devices), *data[1:])
