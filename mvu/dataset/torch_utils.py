from typing import Union, Tuple, Optional

from torch import Tensor
from torch.utils.data import Dataset

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
