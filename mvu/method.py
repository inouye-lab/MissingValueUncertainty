from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Dict

import torch
from overrides import override
from torch import Tensor, Generator

from .dataset import INDEX_SAMPLE, Dataset, INDEX_FEATURE
from .distribution import Distribution
from .imputator import Imputator
from .regressor import Regressor
from .util import estimateResidual


class Method(ABC):
    """Base class defining a method for handling missing values and missing value uncertainty"""

    @abstractmethod
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None) -> Tuple[Tensor, Tensor]:
        """
        Make a prediction using the given features.
        :param features: Input tensor of dimension `(samples, features)` with missingness.
        :param rand:     Random state for random generation
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
        return torch.zeros((features.shape[INDEX_SAMPLE],))

    @override
    def predictWithUncertainty(self, features: Tensor, rand: Generator = None) -> Tuple[Tensor, Tensor]:
        mean = self.regressor.predict(self.imputator.impute(features))
        uncertainty = self.estimateUncertainty(features, rand)
        return mean, uncertainty


C = TypeVar('C')
"""Cache key for the empirical method"""


class EmpiricalUncertaintyMethod(BasicCombinationMethod, ABC, Generic[C]):
    """Estimator that mutates a validation dataset to match the input missingness."""

    dataset: Dataset
    """Validation dataset that is mutated to estimate uncertainty"""

    residual: Tensor
    """Residual uncertainty to cancel out, we just want the change in uncertainy"""

    cache: Dict[C, float]
    """Cache of uncertainty for each cache key."""

    def __init__(self, regressor: Regressor, imputator: Imputator, dataset: Dataset, residual: Tensor):
        super().__init__(regressor, imputator)
        self.dataset = dataset
        self.residual = residual
        self.cache = dict()

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
    def mutate(self, cacheKey: C, rand: Generator = None) -> Dataset:
        """
        Mutates the dataset to look like the given tensor using the cache key.
        :param cacheKey:  Computed cache key from the vector sample
        :param rand:      Random state to allow randomized mutations
        :return:  Mutated dataset, must be a copy of `self.dataset`.
        """
        pass

    @override
    def estimateUncertainty(self, features: Tensor, rand: Generator = None) -> Tensor:
        numSamples = features.shape[INDEX_SAMPLE]
        uncertainty = torch.empty((numSamples,), dtype=torch.float)
        for i in range(numSamples):
            vector = features[i, :]

            # if we have seen this combination before, no need to recalculate
            cacheKey = self.cacheKey(vector)
            if cacheKey in self.cache:
                uncertainty[i] = self.cache[cacheKey]
            else:
                # logging.info(f"Cache Miss for {i} from {cacheKey}")
                # if it's a new combination, need to calculate then cache
                mutated = self.mutate(cacheKey, rand)
                residual = estimateResidual(
                    self.regressor,
                    self.imputator.impute(mutated.features, copy=False),
                    mutated.targets
                )
                uncertainty[i] = residual
                self.cache[cacheKey] = residual.item()

        return uncertainty - self.residual


class EmpiricalUncertaintyByCount(EmpiricalUncertaintyMethod[int]):
    """Empirical uncertainty method that matches the number of missing features"""

    @property
    @override
    def name(self) -> str:
        return f"Empirical By Count - {self.imputator.name}"

    @override
    def cacheKey(self, vector: Tensor) -> int:
        return self.dataset.metadata.countDistinctFeatures(torch.isnan(vector))

    @override
    def mutate(self, cacheKey: int, rand: Generator = None) -> Dataset:
        return self.dataset.dropCount(cacheKey, rand=rand)


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
    def mutate(self, cacheKey: Tuple[bool], rand: Generator = None) -> Dataset:
        return self.dataset.dropSpecified(torch.tensor(cacheKey))


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

    def predictWithUncertainty(self, features: Tensor, rand: Generator = None) -> Tuple[Tensor, Tensor]:
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
