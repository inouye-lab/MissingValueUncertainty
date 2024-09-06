import logging
import os
from abc import ABC, abstractmethod

import torch
from overrides import override
from torch import Tensor, Generator

from ..serializer import loadValue, saveValue


class BatchGenerator(ABC):
    """Base class defining a method for creating a monte carlo batch from a sample."""

    @abstractmethod
    def createBatch(self, image: Tensor, samples: int, index: int = None, rand: Generator = None) -> Tensor:
        """
        Creates a batch of images for the given passed image
        :param image:   Original image
        :param samples: Number of samples to take
        :param index:   Index of the sample, for use in caching results. If none then no cache is possible
        :param rand:    Random state
        :return:  Batch of images based on samples
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of this method for saving in result CSV."""
        pass


class CachingBatchGenerator(BatchGenerator):
    """
    Generator that caches its results to disk, so later samples from the same index get faster results.
    Note this generator has unreliable behavior with seeds, it may be best to use a dedicated
    random state for the generator when working with caching generators.
    """

    generator: BatchGenerator
    """Nested generator to cache contents from"""
    cachePath: str
    """Path to the folder containing the cached batches"""
    cacheMask: Tensor
    """Mask used for the cache, important it matches for the index to be valid"""

    def __init__(self, generator: BatchGenerator, cache_path: str, cache_mask: Tensor):
        self.generator = generator
        self.cachePath = cache_path
        self.cacheMask = cache_mask

        # ensure the sample directory exists
        os.makedirs(self.cachePath, exist_ok=True)
        # ensure the mask in the sample directory matches the mask passed, if not this directory will cause issues
        maskPath = os.path.join(self.cachePath, "mask.pklz")
        if os.path.exists(maskPath):
            directoryMask = loadValue(maskPath, Tensor)
            assert torch.equal(cache_mask.cpu(), directoryMask), \
                f"Directory mask mismatches: passed {self.cacheMask}, but directory contains {directoryMask}"
        else:
            # if the directory lacks a mask, it is probably new, so just save our mask
            saveValue(cache_mask.cpu(), maskPath, Tensor)

    @property
    @override
    def name(self) -> str:
        return f"Caching {self.generator.name}"

    @override
    def createBatch(self, image: Tensor, samples: int, index: int = None, rand: Generator = None) -> Tensor:
        if index is None:
            logging.info("No sample index passed, skipping cache.")
        elif not torch.equal(self.cacheMask, torch.isnan(image)):
            logging.error(f"Image {index} mask does not match the method's mask, unable to use cache")
        else:
            # caching is possible, do we have a cached value?
            batchPath = os.path.join(self.cachePath, f"{index}.pklz")
            if not os.path.exists(batchPath):
                logging.info(f"No cache found for sample {index}, generating new batch.")
            else:
                # cache is valid, use that
                cached = loadValue(batchPath, Tensor).to(image.device)
                if cached.shape[0] >= samples:
                    # note the cache may contain more samples than requested, just take the number requested
                    # means it was created with a different variant of this method
                    # TODO: consider random indexing instead?
                    return cached[0:samples]
                neededSamples = samples - cached.shape[0]
                logging.info(
                    f"Image {index} only has {cached.shape[0]} cached samples, computing {neededSamples} additional samples")
                batch = self.generator.createBatch(image, samples - cached.shape[0], index, rand)
                # combine the two sets and cache the larger number of samples
                batch = torch.cat((cached, batch), dim=0)
                saveValue(batch.cpu(), batchPath, Tensor)
                # no need to index, size is calculated exactly
                return batch
        batch = self.generator.createBatch(image, samples, index, rand)
        saveValue(batch.cpu(), batchPath, Tensor)
        return batch
