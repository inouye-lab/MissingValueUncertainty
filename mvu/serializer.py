import gzip
import pickle
from typing import Any


class SerializerMixin:
    """Generic mixin that adds functions for saving and loading from binary"""

    def save(self, path: str) -> None:
        """
        Saves the data to binary
        :param path: Path excluding extension
        """
        with gzip.open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def _processPostLoad(cls, data: Any) -> "SerializerMixin":
        """
        Method to allow processing the data after loading.
        Useful to wrap related classes in the proper subclass.
        """
        return data

    @classmethod
    def load(cls, path: str) -> "SerializerMixin":
        """
        Loads the splits from binary
        :param path: Path excluding extension
        :return: Loaded dataset
        """
        with gzip.open(path, 'rb') as f:
            data = cls._processPostLoad(pickle.load(f))
        assert isinstance(data, cls)
        return data
