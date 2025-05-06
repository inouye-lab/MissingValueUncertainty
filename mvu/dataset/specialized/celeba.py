import logging
import os
from typing import Tuple, List, Optional, Union

import torch
from overrides import override
from torch import Tensor
from torch.utils.data import Dataset

from ..image import ImagePathDataset
from ..meta import ImageDatasetMeta
from ..torch_utils import TorchDatasetSplits


class CelebAttributes:
    names: List[str]
    """Names of each attribute in the file"""
    originalNames: List[str]

    attributes: Tensor
    """Boolean tensor mapping the index to the value for each key"""

    def __init__(self, path: str, targets: List[str] = None):
        with open(path, 'r') as file:
            # first line is the row count
            rows = int(file.readline().strip())
            # next line is the name of each attribute
            self.names = file.readline().strip().split()
            self.originalNames = self.names
            attributeCount = len(self.names)
            # declare attribute tensor
            self.attributes = torch.empty((rows, attributeCount), dtype=torch.bool)

            seenIndices = torch.zeros(rows, dtype=torch.bool)

            # read in each row
            for _ in range(rows):
                # row starts with the file name, followed by the attributes
                rowValues = file.readline().strip().split()
                assert len(rowValues) == attributeCount + 1, f"Received wrong number of attributes {len(rowValues)}, expected {attributeCount+1}"
                # assuming file names are indices, and file types are jpg
                ext = rowValues[0][-4:]
                assert ext == ".jpg", f"Found invalid file extension {ext}"
                iIdx = int(rowValues[0][:-4])
                # assuming indices are 0 to rows
                assert iIdx < rows, f"Found too large index {iIdx}, expected at most {rows}"

                # ensure we don't parse the same index twice
                assert not seenIndices[iIdx]
                seenIndices[iIdx] = True

                # fill in boolean data for the row, will be 1 when true and -1 when false
                for aIdx in range(attributeCount):
                    self.attributes[iIdx, aIdx] = rowValues[aIdx + 1] == "1"

            # ensure every index was parsed
            for i in range(rows):
                assert seenIndices[i], f"Failed to parse attributes for index {i}"

        if targets is not None:
            assert len(targets) > 0, "Must have at least 1 target"
            # python list.index will automatically validate that the target is actually in teh list
            indices = [self.names.index(target) for target in targets]
            self.attributes = self.attributes[:, indices]
            self.names = targets

    def __len__(self):
        return self.attributes.shape[0]

    def __getitem__(self, item):
        return self.attributes[item]

    def featureIndex(self, name: str) -> int:
        """Gets the index of the given feature by name, or raises a ValueError if its not present"""
        return self.names.index(name)


class CelebADataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset returning the pair of CelebA image and the image attributes"""

    images: ImagePathDataset
    """Dataset for loading images"""
    attributes: CelebAttributes
    """Attributes tensor"""
    indices: List[int]
    """Mapping from a dataset index to a attribute index"""
    returnIndex: bool
    """
    If true, returns the index alongside the image and the attributes.
    The index should generally be considered hidden information from the model,
    however the index may be useful for caches alongside the input data.
    """

    def __init__(self, attributes: CelebAttributes, imagesRoot: str,
                 images: Optional[List[str]] = None, imageList: Optional[str] = None,
                 returnIndex: bool = False):
        self.images = ImagePathDataset(imagesRoot, images, imageList)
        assert len(self.images) > 0, f"Found no images at path {imagesRoot}"
        self.attributes = attributes
        self.returnIndex = returnIndex
        # TODO: generalize this to other file extensions?
        # TODO: move this to image.py?
        self.indices = [int(s[:-4]) for s in self.images.paths]
        samplesWithAttributes = len(self.attributes)
        for idx in self.indices:
            assert idx < samplesWithAttributes

    def __len__(self):
        return len(self.indices)

    @override
    def __getitem__(self, item) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, int]]:
        # image index, used for fetching attribute value
        # note due to splits/shuffling this won't match item
        index = self.indices[item]
        if self.returnIndex:
            return self.images[item], self.attributes[index].float(), index
        return self.images[item], self.attributes[index].float()


def _inRoot(root: Optional[str], folder: Optional[str]) -> Optional[str]:
    if root is None:
        return None
    if folder is None:
        return root
    return os.path.join(root, folder)


def createCelebADataset(attributes_path: str, images_root: str, lists_root: Optional[str] = None,
                        *args,
                        image_size: int = 256, sensor_size: int = 1,
                        train_folder: str = None,      train_list: str = "train_shuffled.flist",
                        validation_folder: str = None, validation_list: str = "val_shuffled.flist",
                        test_folder: str = None,       test_list: str = "test_shuffled.flist",
                        return_index: bool = False, targets: List[str] = None) -> TorchDatasetSplits:
    """Loads in the CelebA dataset using the passed paths"""

    attributes = CelebAttributes(attributes_path, targets=targets)
    logging.info(f"Found {len(attributes.names)} CelebA attributes for {len(attributes)} images: {attributes.names}")
    meta = ImageDatasetMeta("CelebA", attributes.names, image_size, sensor_size, 3)

    # setup image folders
    train = CelebADataset(attributes, _inRoot(images_root, train_folder),
                          imageList=_inRoot(lists_root, train_list), returnIndex=return_index)
    validate = CelebADataset(attributes, _inRoot(images_root, validation_folder),
                             imageList=_inRoot(lists_root, validation_list), returnIndex=return_index)
    test = CelebADataset(attributes, _inRoot(images_root, test_folder),
                         imageList=_inRoot(lists_root, test_list), returnIndex=return_index)
    logging.info(f"Loading {len(train)} training images, {len(validate)} validation images, and {len(test)} testing images")

    return TorchDatasetSplits(train, validate, test, meta)



