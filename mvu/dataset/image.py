import os.path
from typing import List, Optional

from PIL import Image
import torch
from overrides import override
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def load_image(path) -> Tensor:
    """
    Opens the image at the given path as a Torch tensor with pixel values on [-1, 1].
    :param path:  Path to the image.
    :return:  Image loaded as a tensor.
    """
    image = Image.open(path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return (to_tensor(image).to(torch.float32) * 2) - 1


class ImagePathDataset(Dataset[Tensor]):
    """
    Dataset that loads an image Tensor from a list of paths and a root folder.
    """

    root: str
    """Root folder for this dataset"""
    paths: List[str]
    """List of image paths in this dataset"""

    def __init__(self, root: str, paths: Optional[List[str]] = None, fileList: Optional[str] = None):
        """
        Creates the image path dataset
        :param root:       Root folder
        :param paths:      List of paths within the folder for images. If none will load from all folder contents.
        :param fileList:   Path to a file containing the paths list.
        """
        super().__init__()
        self.root = root
        # if not given a list of paths, fetch all paths from the folder
        if paths is not None:
            assert fileList is None, "Cannot set both paths and fileList"
            self.paths = paths
        elif fileList is not None:
            with open(fileList, 'r') as f:
                # can't use f.readlines() as it preserves the trailing newlines, we want to trim them
                # thus its more memory efficient to use the iterator
                self.paths = [line.strip() for line in f]
        else:
            self.paths = []
            for path in os.listdir(root):
                if os.path.exists(os.path.join(root, path)):
                    self.paths.append(path)
            self.paths.sort()

    def __len__(self):
        return len(self.paths)

    @override
    def __getitem__(self, item):
        return load_image(os.path.join(self.root, self.paths[item]))
