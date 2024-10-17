import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
from overrides import override
from torch import Tensor, Generator

from .common import CachableModel, Namable
from .imputator import Imputator
from ..serializer import loadValue, saveValue


class BatchGenerator(CachableModel, Namable, ABC):
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


class CachingBatchGenerator(BatchGenerator):
    """
    Generator that caches its results to disk, so later samples from the same index get faster results.
    Note this generator has unreliable behavior with seeds, it may be best to use a dedicated
    random state for the generator when working with caching generators.
    """

    generator: Optional[BatchGenerator]
    """Nested generator to cache contents from. If none, uses just the cache and throws if no cache value."""
    cachePath: str
    """Path to the folder containing the cached batches"""
    cacheMask: Tensor
    """Mask used for the cache, important it matches for the index to be valid"""

    def __init__(self, generator: Optional[BatchGenerator], cache_path: str, cache_mask: Tensor):
        self.generator = generator
        self.cachePath = cache_path
        self.cacheMask = cache_mask

        # ensure the sample directory exists
        if generator is not None:
            os.makedirs(self.cachePath, exist_ok=True)
        # ensure the mask in the sample directory matches the mask passed, if not this directory will cause issues
        maskPath = os.path.join(self.cachePath, "mask.pklz")
        if os.path.exists(maskPath):
            directoryMask = loadValue(maskPath, Tensor)
            assert torch.equal(cache_mask.cpu(), directoryMask), \
                f"Directory mask mismatches: passed {self.cacheMask}, but directory contains {directoryMask}"
        elif generator is None:
            # if no generator, then an missing mask means we passed the wrong cache path
            raise ValueError(f"No generator cache found at {cache_path}, unable to create without generator")
        else:
            # if the directory lacks a mask, it is probably new, so just save our mask
            saveValue(cache_mask.cpu(), maskPath, Tensor)

    @property
    @override
    def name(self) -> str:
        if self.generator is None:
            return f"Cache from {self.cachePath}"
        return f"Caching {self.generator.name}"

    @override
    def createBatch(self, image: Tensor, samples: int, index: int = None, rand: Generator = None) -> Tensor:
        if index is None:
            assert self.generator is not None, "Must pass index to use caching generator without generator."
            logging.info("No sample index passed, skipping cache.")
        elif not torch.equal(self.cacheMask, torch.isnan(image)):
            assert self.generator is not None, f"Image {index} must match cache mask to use without generator"
            logging.error(f"Image {index} mask does not match the method's mask, unable to use cache")
        else:
            # caching is possible, do we have a cached value?
            batchPath = os.path.join(self.cachePath, f"{index}.pklz")
            if not os.path.exists(batchPath):
                if self.generator is None:
                    raise ValueError(f"No cache found for sample {index}.")
                logging.info(f"No cache found for sample {index}, generating new batch.")
            else:
                # cache is valid, use that
                cached = loadValue(batchPath, Tensor).to(image.device)
                if cached.shape[0] >= samples:
                    # note the cache may contain more samples than requested, just take the number requested
                    # means it was created with a different variant of this method
                    # TODO: consider random indexing instead?
                    return cached[0:samples]
                # if unable to generate, throw
                if self.generator is None:
                    raise ValueError(f"Requested {samples} samples for {index}, but cache only contains {cached.shape[0]}")

                neededSamples = samples - cached.shape[0]
                logging.info(f"Image {index} only has {cached.shape[0]} cached samples, "
                             f"computing {neededSamples} additional samples")
                batch = self.generator.createBatch(image, samples - cached.shape[0], index, rand)
                # combine the two sets and cache the larger number of samples
                batch = torch.cat((cached, batch), dim=0)
                saveValue(batch.cpu(), batchPath, Tensor)
                # no need to index, size is calculated exactly
                return batch
        batch = self.generator.createBatch(image, samples, index, rand)
        saveValue(batch.cpu(), batchPath, Tensor)
        return batch

    def hasCache(self, index) -> bool:
        """
        Checks if the generator has a cache for the given index
        :param index:   Index to check.
        :return:   True if the index is already cached
        """
        return os.path.exists(os.path.join(self.cachePath, f"{index}.pklz"))

    @override
    def supportsIndices(self, indices: Tensor) -> Tensor:
        if self.generator is not None:
            return torch.ones_like(indices, dtype=torch.bool)
        supports = torch.zeros_like(indices, dtype=torch.bool)
        for i, index in enumerate(indices):
            supports[i] = self.hasCache(index.item())
        return supports


class SingleSampleImputator(Imputator):
    """
    Imputator that takes a single sample from a batch generator as the imputation.
    """

    generator: BatchGenerator
    """Generator instance for gathering single sample estimations"""

    def __init__(self, generator: BatchGenerator):
        self.generator = generator

    @property
    @override
    def name(self) -> str:
        return f"Single Sample {self.generator.name} Imputation"

    def _impute(self, features: Tensor, rand: Generator = None, indices: Tensor = None) -> None:
        for i in range(features.shape[0]):
            index = None if indices is None else indices[i]
            features[i] = self.generator.createBatch(features, 1, index=index, rand=rand)

    @override
    def supportsIndices(self, indices: Tensor) -> Tensor:
        return self.generator.supportsIndices(indices)
