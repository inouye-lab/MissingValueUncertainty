from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Dict, Optional

import torch
from overrides import override
from torch import Tensor, Generator
from torch.utils.data import DataLoader

from .distribution import Distribution
from .generator import BatchGenerator
from .imputator import Imputator
from .regressor import Regressor
from ..dataset.csv import CsvDataset
from ..dataset.meta import INDEX_SAMPLE, INDEX_FEATURE, DatasetMeta


class Method(ABC):
    """Base class defining a method for handling missing values and missing value uncertainty"""

    @abstractmethod
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None, index: int = None
                               ) -> Tuple[Tensor, Tensor]:
        """
        Make a prediction using the given features.
        :param features: Input tensor of dimension `(samples, features)` with missingness.
        :param rand:     Random state for random generation
        :param index:    Sample index for the sake of caching. This should only be used to reduce computation times,
                         not in any way that provides access to normally hidden data.
        :return: Vector of prediction means `(samples,)` and missing value variances `(samples,)`
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of this method for saving in result CSV."""
        pass


class BasicCombinationMethod(Method):
    """
    Method that combines a regressor and an imputator to make predictions.
    Default method of handling uncertainty just returns zero, but subclasses may use a more intelligent method.
    """

    regressor: Regressor
    imputator: Imputator

    def __init__(self, regressor: Regressor, imputator: Imputator):
        self.regressor = regressor
        self.imputator = imputator

    @property
    @override
    def name(self) -> str:
        return f"Basic Imputation - {self.imputator.name}"

    def estimateUncertainty(self, features: Tensor, rand: Generator = None) -> Tensor:
        """
        Estimate the missing value uncertainty in the prediction. Default implementation just returns zero
        :param features: Input tensor of dimension `(samples, features)` with missingness.
        :param rand:     Random state for random generation
        :return: Vector of missing value variances of size `(samples,)`
        """
        return torch.zeros((features.shape[INDEX_SAMPLE],), device=features.device)

    @override
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None, index: int = None
                               ) -> Tuple[Tensor, Tensor]:
        mean = self.regressor.predict(self.imputator.impute(features))
        uncertainty = self.estimateUncertainty(features, rand)
        return mean, uncertainty


C = TypeVar('C')
"""Cache key for the empirical method"""


class EmpiricalUncertaintyMethod(BasicCombinationMethod, ABC, Generic[C]):
    """Estimator that mutates a validation dataset to match the input missingness."""

    metadata: DatasetMeta
    """Metadata from the dataset to remove features"""

    data: DataLoader
    """Validation dataset via data loader that is mutated to estimate uncertainty"""

    residual: Tensor
    """Residual uncertainty to cancel out as a tensor of size 1, we just want the change in uncertainly"""

    cache: Dict[C, float]
    """Cache of uncertainty for each cache key."""

    def __init__(self, regressor: Regressor, imputator: Imputator,
                 metadata: DatasetMeta, data: DataLoader, residual: Tensor):
        super().__init__(regressor, imputator)
        self.metadata = metadata
        self.data = data
        self.residual = residual
        self.cache = dict()

    @classmethod
    def fromDataset(cls, regressor: Regressor, imputator: Imputator, dataset: CsvDataset, residual: Tensor
                    ) -> "EmpiricalUncertaintyMethod":
        """Creates an instance of this method using a dataset, with automatically set batch size."""
        return cls(regressor, imputator, dataset.metadata,
                   DataLoader(dataset.toTorch(), batch_size=100, shuffle=False), residual)

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def cacheKey(self, vector: Tensor) -> C:
        """
        Converts the given vector into a cache key. Will attempt lookup, and if that fails compute.
        :param vector: Input sample
        :return:  Cache key, must be enough to mutate the dataset to estimate uncertainty.
        """
        pass

    @abstractmethod
    def mutate(self, features: Tensor, cacheKey: C, rand: Generator = None) -> Tensor:
        """
        Mutates the dataset to look like the given tensor using the cache key.
        :param features:  Features to mutate
        :param cacheKey:  Computed cache key from the vector sample
        :param rand:      Random state to allow randomized mutations
        :return:  Mutated tensor, must be a copy of `features`.
        """
        pass

    @override
    def estimateUncertainty(self, features: Tensor, rand: Generator = None) -> Tensor:
        numSamples = features.shape[INDEX_SAMPLE]
        uncertainty = torch.empty((numSamples,), device=features.device, dtype=torch.float)
        for i in range(numSamples):
            vector = features[i, :]

            # if we have seen this combination before, no need to recalculate
            cacheKey = self.cacheKey(vector)
            if cacheKey in self.cache:
                uncertainty[i] = self.cache[cacheKey]
            else:
                # logging.info(f"Cache Miss for {i} from {cacheKey}")
                # if it's a new combination, need to calculate then cache
                # calculate squared error over time to prevent bias from the specific samples
                device = features.device
                squaredError = torch.tensor([0], device=device, dtype=torch.float)
                seenSamples = 0

                # simply process each batch one at a time, no need to do anything fancy with loaders
                for (validateFeatures, validateTargets) in self.data:
                    validateFeatures = validateFeatures.to(device)
                    validateTargets = validateTargets.to(device)
                    means = self.regressor.predict(
                        self.imputator.impute(
                            self.mutate(validateFeatures, cacheKey, rand),
                            copy=False
                        )
                    )
                    squaredError += ((means - validateTargets) ** 2).sum()
                    seenSamples += validateTargets.shape[0]
                assert seenSamples != 0, "No samples in empirical uncertainty method"
                mse = squaredError / seenSamples
                uncertainty[i] = mse
                self.cache[cacheKey] = mse.item()

        return uncertainty - self.residual


class EmpiricalUncertaintyByCount(EmpiricalUncertaintyMethod[int]):
    """Empirical uncertainty method that matches the number of missing features"""

    @property
    @override
    def name(self) -> str:
        return f"Empirical By Count - {self.imputator.name}"

    @override
    def cacheKey(self, vector: Tensor) -> int:
        return self.metadata.countDistinctFeatures(torch.isnan(vector))

    @override
    def mutate(self, features: Tensor, cacheKey: int, rand: Generator = None) -> Tensor:
        return self.metadata.dropCount(features, cacheKey, rand=rand)


class EmpiricalUncertaintyByFeature(EmpiricalUncertaintyMethod[Tuple[bool]]):
    """
    Empirical uncertainty method that matches the specific missing features.
    Key is a boolean tensor of features to remove of size `(features,)`
    """

    @property
    @override
    def name(self) -> str:
        return f"Empirical By Feature - {self.imputator.name}"

    @override
    def cacheKey(self, vector: Tensor) -> Tuple[bool]:
        return tuple(torch.isnan(vector).tolist())

    @override
    def mutate(self, features: Tensor, cacheKey: Tuple[bool], rand: Generator = None) -> Tensor:
        return self.metadata.dropSpecified(features, torch.tensor(cacheKey))


class MonteCarloMethod(Method):
    """Method that takes a number of samples from a distribution then aggregates the results"""

    regressor: Regressor
    distribution: Distribution
    samples: int
    """Number of Monte Carlo samples to take"""

    def __init__(self, regressor: Regressor, distribution: Distribution, samples: int):
        self.regressor = regressor
        self.distribution = distribution
        self.samples = samples

    @property
    @override
    def name(self) -> str:
        return f"Monte Carlo - {self.distribution.name} - {self.samples} samples"

    def predictWithUncertainty(self, features: Tensor, rand: Generator = None, index: int = None
                               ) -> Tuple[Tensor, Tensor]:
        dataSamples = features.shape[INDEX_SAMPLE]
        numFeatures = features.shape[INDEX_FEATURE]

        # start by computing augmented data with samples
        augmented = self.distribution.augment(features, self.samples, rand)

        # reshape that input into a tensor of `(samples, features)`
        regressorInput = augmented.reshape((dataSamples*self.samples, numFeatures))

        # predict result and reshape into matrix of `(dataSample, distSample)` for summarizing
        predictions = self.regressor.predict(regressorInput).reshape((dataSamples, self.samples))

        # finally, return our two results
        return torch.mean(predictions, dim=1).reshape(-1), \
            torch.var(predictions, dim=1).reshape(-1)


class MonteCarloBatchMethod(Method):
    regressor: Regressor
    generator: BatchGenerator
    samples: int
    """Number of Monte Carlo samples to take"""

    def __init__(self, regressor: Regressor, generator: BatchGenerator, samples: int):
        self.regressor = regressor
        self.generator = generator
        self.samples = samples

    @property
    @override
    def name(self) -> str:
        return f"Monte Carlo - {self.generator.name} - {self.samples} samples"

    @override
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None, index: int = None
                               ) -> Tuple[Tensor, Tensor]:
        featureSamples = features.shape[INDEX_SAMPLE]
        means: Optional[Tensor] = None
        variances: Optional[Tensor] = None

        for fIdx in range(features.shape[INDEX_SAMPLE]):
            batch = self.generator.createBatch(features[fIdx], self.samples, index, rand)
            prediction = self.regressor.predict(batch)
            # we don't know the output size without running the regressor, so lazily init the output tensors
            if means is None:
                means = torch.empty((featureSamples, *prediction.shape[1:]), device=features.device)
            if variances is None:
                variances = torch.empty((featureSamples, *prediction.shape[1:]), device=features.device)
            # fill in output from the prediction
            means[fIdx, :] = prediction.mean(dim=0)
            variances[fIdx, :] = prediction.var(dim=0)

        return means, variances
