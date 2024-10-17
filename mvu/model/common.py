from abc import ABC, abstractmethod

import torch
from torch import Tensor


class CachableModel(ABC):
    """
    Declares that a method supports caching, and thus needs a way to check if a sample is usable when only caching.
    """

    def supportsIndices(self, indices: Tensor) -> Tensor:
        """
        Checks if the given method supports the given indices.
        :param indices:  List of indices to check
        :return:  Boolean tensor where true indices the given index is supported
        """
        return torch.ones_like(indices, dtype=torch.bool)


class Namable(ABC):
    """Common interface of objects with names"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of this object for saving in result CSV."""
        pass
