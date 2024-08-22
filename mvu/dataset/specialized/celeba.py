import logging
import os
from typing import Tuple, List

import torch
from overrides import override
from torch import Tensor
from torch.utils.data import Dataset

from ..image import ImagePathDataset
from ..meta import ImageDatasetMeta
from ..torch import TorchDatasetSplits


class CelebAttributes:
    names: List[str]
    """Names of each attribute in the file"""

    attributes: Tensor
    """Boolean tensor mapping the index to the value for each key"""

    def __init__(self, path: str):
        with open(path, 'r') as file:
            # first line is the row count
            rows = int(file.readline().strip())
            # next line is the name of each attribute
            self.names = file.readline().strip().split()
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

    def __len__(self):
        return self.attributes.shape[0]

    def __getitem__(self, item):
        return self.attributes[item]


class CelebADataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset returning the pair of CelebA image and the image attributes"""

    images: ImagePathDataset
    """Dataset for loading images"""
    attributes: CelebAttributes
    """Attributes tensor"""
    indices: List[int]
    """Mapping from a dataset index to a attrbiute index"""

    def __init__(self, attributes: CelebAttributes, imagesRoot: str, images: List[str] = None):
        self.images = ImagePathDataset(imagesRoot, images)
        assert len(self.images) > 0, f"Found no images at path {imagesRoot}"
        self.attributes = attributes
        self.indices = [int(s[:-4]) for s in self.images.paths]
        samplesWithAttributes = len(self.attributes)
        for idx in self.indices:
            assert idx < samplesWithAttributes

    def __len__(self):
        return len(self.indices)

    @override
    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        return self.images[item], self.attributes[self.indices[item]].float()


def createCelebADataset(attributes_path: str, images_root: str,
                        train_folder: str, validation_folder: str, test_folder: str,
                        image_size: int = 256, sensor_size: int = 1) -> TorchDatasetSplits:
    """Loads in the CelebA dataset using the passed paths"""

    attributes = CelebAttributes(attributes_path)
    logging.info(f"Found {len(attributes.names)} CelebA attributes for {len(attributes)} images: {attributes.names}")
    # TODO: support filtering down targets to a smaller list
    meta = ImageDatasetMeta("CelebA", attributes.names, image_size, sensor_size, 3)

    # setup image folders
    train = CelebADataset(attributes, os.path.join(images_root, train_folder))
    validate = CelebADataset(attributes, os.path.join(images_root, validation_folder))
    test = CelebADataset(attributes, os.path.join(images_root, test_folder))
    logging.info(f"Loading {len(train)} training images, {len(validate)} validation images, and {len(test)} testing images")

    return TorchDatasetSplits(train, validate, test, meta)



