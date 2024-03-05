import logging
from abc import ABC, abstractmethod
import threading
from typing import Union, Tuple, Optional

import torch
from overrides import override
from torch import Tensor, Generator
# from torch.distributions.multivariate_normal import MultivariateNormal

from numpy import random
from torch.utils.data import DataLoader

from .imputator import containsMissing, Imputator
from ..dataset.csv import CsvDataset
from ..dataset.meta import validateFeatures, INDEX_SAMPLE, INDEX_FEATURE, DatasetMeta
from ..serializer import SerializerMixin


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
        augmentedFeatures = torch.zeros((dataSample, distSamples, numFeatures),
                                        device=features.device, dtype=torch.float)

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


class GaussianParameters(SerializerMixin):
    """Represents the learnable parameters of a gaussian distribution"""

    mean: Tensor
    """Mean vector of size `(features,)`"""
    covariance: Tensor
    """Covariance matrix of size `(features,features)`"""

    def __init__(self, mean: Tensor, covariance: Tensor):
        assert covariance.shape[0] == covariance.shape[1], "Covariance matrix must be square"
        assert mean.shape[0] == covariance.shape[0], "Covariance matrix must have same number of features as mean"
        self.mean = mean
        self.covariance = covariance
        if not torch.eq(covariance, covariance.T).all():
            logging.warning("Covariance matrix is not symmetric")

    @classmethod
    def fromCsvDataset(cls, dataset: CsvDataset):
        """Creates an instance from a CSV dataset (comes with full samples in a single tensor)"""
        return cls(
            torch.mean(dataset.features, dim=INDEX_SAMPLE),
            torch.cov(dataset.features.T)
        )

    @classmethod
    def fromDataloader(cls, numInputs: int, data: DataLoader, showProgress: bool = False, device: torch.device = None):
        """
        Creates an instance from a data loader (requires processing samples in batches)
        :param numInputs:    Number of input features
        :param data:         Data loader to use in creating the parameters
        :param showProgress: If true, prints regular updates on progress
        :param device:       Device to use for computing the mean and variance, and the resulting parameter device
        :return  Distribution instance on the passed device
        """
        # need the means to compute the covariances
        numSamples = 0
        means = torch.zeros((numInputs,), device=device)
        batches = len(data)
        for i, (features, targets) in enumerate(data):
            if showProgress:
                print(f"Computing mean batch {i+1:10}/{batches}", end="\r")
            if device is not None:
                features = features.to(device)
            means += features.sum(axis=0)
            numSamples += features.shape[0]
        means /= numSamples
        logging.info(f"Computed gaussian mean")

        # unfortunately have to compute the covariance in a less optimal way as we must do it over a dataloader
        # TODO: reconsider method of features*features/n - mean*mean, but using outer product method (matmul did inner)
        covariance = torch.zeros((numInputs, numInputs), device=device)
        for i, (features, targets) in enumerate(data):
            if showProgress:
                print(f"Computing covariance batch {i+1:10}/{batches}", end="\r")
            if device is not None:
                features = features.to(device)
            diffVector = features - means
            covariance += torch.matmul(diffVector.T, diffVector)
        covariance /= numSamples
        logging.info(f"Computed gaussian covariance")

        # create the final distribution
        return cls(means, covariance)

    def to(self, device: torch.device) -> "GaussianParameters":
        """
        Copies the parameters to the given device
        :param device:  Device to use
        :return:  New instance of params on the new device
        """
        return GaussianParameters(self.mean.to(device), self.covariance.to(device))

    def cpu(self) -> "GaussianParameters":
        """Moves this parameter set to the CPU"""
        return GaussianParameters(self.mean.cpu(), self.covariance.cpu())


class MarginalGaussianDistribution(Imputator, Distribution):
    """Distribution implementing a marginalized gaussian."""

    datasetMeta: DatasetMeta
    """Metadata of the dataset for normalization"""

    params: GaussianParameters
    """Distribution parameters"""

    _local: threading.local
    """
    Storage in the local thread for a hack workaround to lack of nice method for gaussian sampling in torch
    Should swap this from class storage to thread local instance in the future
    Temporary generator used during sample generation as we are unable to use the torch multivariate normal
    """

    def __init__(self, datasetMeta: DatasetMeta, params: GaussianParameters):
        datasetMeta.validateFeatures(params.mean, isVector=True)
        self.datasetMeta = datasetMeta
        self.params = params
        self._local = threading.local()

    @property
    @override
    def name(self) -> str:
        return "Marginal Gaussian"

    @property
    def mean(self) -> Tensor:
        return self.params.mean

    @property
    def covariance(self) -> Tensor:
        return self.params.covariance

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
            empty = torch.tensor([], device=vector.device)
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
        self._local.generator = random.default_rng(torch.randint(2**32-1, (1,), generator=rand).item())
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
        return torch.tensor(
            self._local.generator.multivariate_normal(missingMean.cpu().numpy(), missingCov.cpu().numpy(), distSamples),
            device=sample.device, dtype=torch.float
        )

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

    leastSquares: bool
    """
    If true, uses the least squares approach for operations involving the inverse of the observed covariances.
    If false, uses the pseudo inverse.
    """
    covarianceInv: Optional[Tensor]
    """
    If not None, computes the covariance using the schur complement of the observed covariances.
    If None, computes the covariance using matrix multiplications with the inverse of the observed covariances.
    """
    hermitian: bool
    """
    If true, matrix inversions use eigen value decomposition using the lower triangle for inverses.
    If false, matrix inverses use singular value decomposition.
    Unused if `leastSquares` is True and not using `schur`.
    """

    def __init__(self, datasetMeta: DatasetMeta, params: GaussianParameters,
                 schur: Union[Tensor, bool] = False, leastSquares: bool = True, hermitian: bool = True):

        super().__init__(datasetMeta, params)
        self.leastSquares = leastSquares
        self.hermitian = hermitian

        # if we are using schur, we want the inverted covariance matrix, faster to compute just once
        # noinspection PySimplifyBooleanCheck
        if schur == True:
            self.covarianceInv = torch.linalg.pinv(self.covariance, hermitian=hermitian)
        elif schur == False:
            self.covarianceInv = None
        else:
            assert schur.shape == self.covariance.shape, "Covariance inverse matrix must be square and of input size"
            self.covarianceInv = schur

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

        # decide whether to use the least squares approach or the pseudo inverse to multiply observed covariances
        # by the mean & observation difference

        # observed partition of the covariances
        # nonzero returns column vectors, so we need to transpose the second to treat as a row vector
        obsCov = self.covariance[observedIndices, observedIndices.T]
        # offset of the observations from the mean
        obsOffset = vector[observedMask] - self.mean[observedMask]
        # inverse of the observed covariances, None if self.leastSquares
        obsCovInv: Optional[Tensor]
        # multiplication of obsCovInv and obsOffset
        scaledOffset: Tensor
        if self.leastSquares:
            # least squares does not work on CUDA using the desired algorithm, the only supported CUDA leads to NaNs
            # to work around this, just move to the CPU for this calculation, slower but its the best option
            # TODO: consider if this is dataset specific
            # TODO: perhaps perform this entire method on the CPU?
            scaledOffset = torch.linalg.lstsq(obsCov.cpu(), obsOffset.cpu(), driver="gelsy").solution.to(vector.device)
        else:
            obsCovInv = torch.linalg.pinv(obsCov, hermitian=self.hermitian)
            scaledOffset = torch.matmul(obsCovInv, obsOffset)
        # Partition of covariance containing covariances between missing indexes and observed indexes
        corrMatrix = self.covariance[missingIndices, observedIndices.T]
        # Final computed conditional mean
        condMean = self.mean[missingMask] + torch.matmul(corrMatrix, scaledOffset)

        # Quick exit if we do not care about the conditional covariance
        if not returnCovariance:
            return condMean

        # Final computed conditional variance
        if self.covarianceInv is not None:
            # compute the covariance using the schur complement,
            # this tends to be more stable than matrix multiplications but can be slower due to the extra inverse
            condVar = torch.linalg.pinv(self.covarianceInv[missingIndices, missingIndices.T], hermitian=self.hermitian)
        else:
            # compute the covariance using an offset from the unobserved covariances,
            # requires more matrix multiplications but fewer inverses,
            # can potentially lead to a crash when not using the least squares approach on some datasets
            # multiplication of obsCovInv and corrMatrix.T
            scaledCorr: Tensor
            if self.leastSquares:
                # see above comment on lstsq algorithms
                scaledCorr = torch.linalg.lstsq(obsCov.cpu(), corrMatrix.T.cpu(), driver="gelsy").solution.to(vector.device)
            else:
                scaledCorr = torch.matmul(obsCovInv, corrMatrix.T)
            condVar = self.covariance[missingIndices, missingIndices.T] - torch.matmul(corrMatrix, scaledCorr)

        # ensure symmetry of covariance matrix
        if not torch.eq(condVar, condVar.T).all():
            # logging.warning(f"Covariance matrix symmetry is off by {torch.sum(torch.abs(condVar - condVar.T))}")
            condVar = (condVar + condVar.T) / 2

        return condMean, condVar
