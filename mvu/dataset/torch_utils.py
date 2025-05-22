import logging
from typing import Union, Tuple, Optional, Dict

from torch import Tensor
from torch.utils.data import Dataset, Subset

from mvu.dataset.meta import DatasetMeta

TwoTensor = Union[Tuple[Tensor, ...], Tuple[Tensor, Tensor]]
"""Represents the types inside the two different valid forms of a torch dataset"""

TwoTensorDataset = Union[Dataset[Tuple[Tensor, ...]], Dataset[Tuple[Tensor, Tensor]]]
"""Represents the two different valid forms of a torch dataset"""


class TorchDatasetSplits:
    """Object representing a train, validation, and testing split on a torch dataset"""

    train: TwoTensorDataset
    """Dataset used for learning the model"""

    validate: TwoTensorDataset
    """Dataset used for learning hyperparameters"""

    test: TwoTensorDataset
    """Dataset used for validating results"""

    metadata: Optional[DatasetMeta]
    """Metadata in the dataset"""

    def __init__(self, train: Dataset[TwoTensor], validate: Dataset[TwoTensor], test: Dataset[TwoTensor],
                 metadata: Optional[DatasetMeta] = None):
        self.train = train
        self.validate = validate
        self.test = test
        self.metadata = metadata


    @staticmethod
    def split(meta: DatasetMeta, trainingValidation: Dataset, testing: Dataset, validation_percent: float = 0.3, samples: Dict[str,int] = None) -> "TorchDatasetSplits":
        """
        Shared code between many dataset types for splitting a dataset
        :param meta:                Dataset meta for the result
        :param trainingValidation:  Dataset to split between training and validation
        :param testing:             Dataset to use for testing
        :param validation_percent:  Percentage of training data to use for validation
        :param samples:             Maximum samples from the dataset to use for train, validate, and test.
        :return:  Dataset instance
        """
        trainingValidationSize = len(trainingValidation)
        validationEnd = int(trainingValidationSize * validation_percent)
        trainingStart = validationEnd
        trainingEnd = trainingValidationSize
        if samples is not None:
            # training has a start offset and an end, so need some math to convert the limit
            if "train" in samples:
                trainLimit = samples["test"]
                if trainLimit < trainingEnd - trainingStart:
                    trainingEnd = trainingStart + trainLimit
            # validate is easy to limit, just reduce max samples
            if "validate" in samples:
                validationEnd = min(validationEnd, samples["validate"])
            # only make test a subset if needed, can use the raw dataset otherwise
            if "test" in samples:
                testLimit = samples["test"]
                if testLimit < len(testing):
                    testing = Subset(testing, range(0, testLimit))
        # apply the computed limits
        training = Subset(trainingValidation, range(trainingStart, trainingEnd))
        validation = Subset(trainingValidation, range(0, validationEnd))

        logging.info(f"Split training from {trainingValidationSize} into {len(training)} training images and {len(validation)} validation images. Found {len(testing)} testing images.")

        return TorchDatasetSplits(training, validation, testing, meta)