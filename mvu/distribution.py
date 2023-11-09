from abc import ABC, abstractmethod
import threading
from typing import Optional, Union, Tuple

import torch
from overrides import override
from torch import Tensor, Generator
# from torch.distributions.multivariate_normal import MultivariateNormal

from numpy import random

from .dataset import INDEX_SAMPLE, INDEX_FEATURE, DatasetMeta, Dataset, validateFeatures
from .imputator import containsMissing, Imputator


class Distribution(ABC):
    """
    Class representing a distribution for the sake of Monte Carlo Methods.
    Implementers will also often implement `Imputator` to support distribution based imputation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of this distribution for saving in result CSV."""
        pass

    def augment(self, features: Tensor, distSamples: int, rand: Generator = None) -> Tensor:
        """
        Creates an augmented for use in Monte Carlo methods.
        :param features:      Input tensor with missing values, size `(dataSamples, features)`
        :param distSamples:   Number of samples of the distribution to take
        :param rand:          Random state
        :return: Matrix of size `(dataSamples, distSamples, features)`.
        """
        self._validateFeatures(features)

        # the goal here is to process all test samples and all input samples in one large batch of size test * input
        # start by constructing a 3D matrix of test sample * input sample * feature
        dataSample = features.shape[INDEX_SAMPLE]
        numFeatures = features.shape[INDEX_FEATURE]
        augmentedFeatures = torch.zeros((dataSample, distSamples, numFeatures))

        # form matrix from samples
        for sampleIndex in range(dataSample):
            sample = features[sampleIndex, :]
            missingIndexes = torch.isnan(sample)

            augmentedFeatures[sampleIndex, :, :] = sample.reshape(1, 1, -1).expand(-1, distSamples, -1)
            # If no missing indexes, no need to handle samples
            if torch.count_nonzero(missingIndexes) > 0:
                augmentedFeatures[sampleIndex, :, missingIndexes] = self._sampleDistribution(sample, distSamples, rand)

                # we should have filled in all nan values in the final array
                assert not containsMissing(augmentedFeatures[sampleIndex, :, :])

        return self._normalize(augmentedFeatures)

    @abstractmethod
    def _validateFeatures(self, features: Tensor) -> None:
        """
        Validate that the features tensor is a supported size
        :param features: Input tensor with missing values, size `(dataSamples, features)`
        """
        pass

    @abstractmethod
    def _sampleDistribution(self, sample: Tensor, distSamples: int, rand: Generator = None) -> Tensor:
        """
        Samples the distribution, producing a matrix of samples.
        :param sample:       Given sample, of size `(features,)`.
        :param distSamples:  Number of samples of this distribution to take
        :param rand:         Random state
        :return:  Matrix of size `(distSamples,features)`.
        """
        pass

    def _normalize(self, augmentedFeatures: Tensor) -> Tensor:
        """
        Normalizes the matrix of augmented features to ensure valid onehot vectors
        :param augmentedFeatures:
        :return:
        """
        return augmentedFeatures


class MarginalGaussianDistribution(Imputator, Distribution):
    """Distribution implementing a marginalized gaussian."""

    datasetMeta: DatasetMeta
    """Metadata of the dataset for normalization"""

    mean: Tensor
    """Mean vector of size `(features,)`"""
    covariance: Tensor
    """Covariance matrix of size `(features,features)`"""

    _local: threading.local
    """
    Storage in the local thread for a hack workaround to lack of nice method for gaussian sampling in torch
    Should swap this from class storage to thread local instance in the future
    Temporary generator used during sample generation as we are unable to use the torch multivariate normal
    """

    def __init__(self, datasetMeta: DatasetMeta, mean: Tensor, covariance: Tensor):
        datasetMeta.validateFeatures(mean, isVector=True)
        assert covariance.shape[0] == covariance.shape[1] == datasetMeta.numInputs, \
            "Covariance matrix must be square and of input size"
        self.datasetMeta = datasetMeta
        self.mean = mean
        self.covariance = covariance
        self._local = threading.local()

    @property
    @override
    def name(self) -> str:
        return "Marginal Gaussian"

    @classmethod
    def fromDataset(cls, dataset: Dataset):
        return cls(
            dataset.metadata,
            torch.mean(dataset.features, dim=INDEX_SAMPLE),
            torch.cov(dataset.features.T)
        )

    @override
    def _validateFeatures(self, features: Tensor) -> None:
        self.datasetMeta.validateFeatures(features)

    def condition(self, vector: Tensor, returnCovariance: bool = True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Conditions or marginalizes the mean vector and covariance matrix on the input vector
        :param vector:            Vector of size `(features,)`
        :param returnCovariance:  If true, returns the covariance matrix. If false, just return the mean vector.
        :return:  Mean vector of size `(features,)` and covariance matrix of size `(features, features)`
        """
        self.datasetMeta.validateFeatures(vector, isVector=True)

        # if no missing indexes, nothing to do
        missingMask = torch.isnan(vector)
        missingCount = torch.count_nonzero(missingMask)
        if missingCount == 0:
            empty = torch.Tensor([])
            if returnCovariance:
                return empty, empty
            return empty

        # if everything is missing, just use the full mean/covariance
        if missingCount == len(vector):
            if returnCovariance:
                return self.mean, self.covariance
            return self.mean

        return self._condition(vector, missingMask, returnCovariance)

    def _condition(self, vector: Tensor, missingMask: Tensor, returnCovariance: bool = True):
        """
        Conditions or marginalizes the mean vector and covariance matrix on the input vector
        :param vector:            Vector of size `(features,)`
        :param missingMask:       Boolean vector of size `(features,)` of missing values
        :param returnCovariance:  If true, returns the covariance matrix. If false, just return the mean vector.
        :return:  Mean vector of size `(features,)` and covariance matrix of size `(features, features)`
        """
        mean = self.mean[missingMask]
        if returnCovariance:
            missingIndices = missingMask.nonzero()
            return mean, self.covariance[missingIndices, missingIndices.T]
        return mean

    @override
    def _impute(self, features: Tensor) -> None:
        validateFeatures(features, len(self.mean))

        for i in range(features.shape[INDEX_SAMPLE]):
            image = features[i, :]
            missingIndexes = torch.isnan(image)
            if torch.count_nonzero(missingIndexes) > 0:
                features[i, missingIndexes] = self.condition(image, returnCovariance=False)
        self.datasetMeta.normalizeFeatures(features, copy=False)

    @override
    def augment(self, features: Tensor, distSamples: int, rand: Generator = None) -> Tensor:
        self._local.generator = random.default_rng(torch.randint(2**32-1, (1,)).item())
        result = super().augment(features, distSamples, rand)
        self._local.generator = None
        return result

    @override
    def _sampleDistribution(self, sample: Tensor, distSamples: int, rand: Generator = None) -> Tensor:
        missingMean, missingCov = self.condition(sample, returnCovariance=True)
        # return MultivariateNormal(missingMean, covariance_matrix=missingCov).sample(torch.Size((distSamples,)))

        # torch requires positive definite instead of positive-semidefinite, so stuck using numpy here
        # ideally we would always have positive definite,
        # but something about the covariance conditioning does not guarantee that
        return torch.Tensor(self._local.generator.multivariate_normal(missingMean, missingCov, distSamples))

    @override
    def _normalize(self, augmentedFeatures: Tensor) -> Tensor:
        dataSamples = augmentedFeatures.shape[0]
        distSamples = augmentedFeatures.shape[1]
        features = augmentedFeatures.shape[2]
        return self.datasetMeta.normalizeFeatures(
            augmentedFeatures.reshape((dataSamples*distSamples, features)), copy=False
        ).reshape(dataSamples, distSamples, features)


class ConditionalGaussianDistribution(MarginalGaussianDistribution):
    """Distribution implementing a conditional gaussian distribution."""

    covarianceInv: Tensor
    """Inverted covariance matrix of size `(features,features)`, speeds up some calculations"""

    def __init__(self, datasetMeta: DatasetMeta, mean: Tensor, covariance: Tensor, covarianceInv: Tensor = None):
        super().__init__(datasetMeta, mean, covariance)
        assert covarianceInv is None or covarianceInv.shape == covariance.shape, \
            "Covariance inverse matrix must be square and of input size"
        if covarianceInv is None:
            self.covarianceInv = torch.linalg.pinv(covariance)
        else:
            self.covarianceInv = covarianceInv

    @property
    @override
    def name(self) -> str:
        return "Conditional Gaussian"

    @override
    def _condition(self, vector: Tensor, missingMask: Tensor, returnCovariance: bool = True):
        # will be partitioning the matrix into missing and observed for following operations
        observedMask = torch.logical_not(missingMask)
        observedIndices = observedMask.nonzero()
        missingIndices = missingMask.nonzero()

        # Partition of covariance containing covariances between missing indexes and observed indexes
        # nonzero returns column vectors, so we need to transpose the second to treat as a row vector
        corrMatrix = self.covariance[missingIndices, observedIndices.T]
        # Inverted partition of covariance containing just observed indexes
        obsCovInv = torch.linalg.pinv(self.covariance[observedIndices, observedIndices.T])
        # Final computed conditional mean
        condMean = self.mean[missingMask] + torch.matmul(
            torch.matmul(corrMatrix, obsCovInv),
            vector[observedMask] - self.mean[observedMask]
        )
        # Quick exit if we do not care about the conditional covariance
        if not returnCovariance:
            return condMean

        # Final computed conditional variance
        # The following formula is more efficient (no need to compute an additional inverse)
        # but leads to a crash on some datasets as it produces a matrix that is not positive semi-definiate
        # condVar = covariance[missingIndexes, missingIndexes] \
        #     - torch.matmul(torch.matmul(corrMatrix, obsCovInv), corrMatrix.T)

        # this formula is mathematically equivalent, but is less efficient
        # however, we should be guaranteed a valid covariance matrix at the end
        condVar = torch.linalg.pinv(self.covarianceInv[missingIndices, missingIndices.T])
        return condMean, condVar
