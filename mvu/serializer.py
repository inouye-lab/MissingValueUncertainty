import gzip
import pickle
from typing import Any, Type, TypeVar


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


def saveValue(data: Any, path: str, cls: Type = None) -> None:
    """
    Saves the data to the given path
    :param data:  Data to save
    :param path:  Path to save the data
    :param cls:   Type of data to save
    """
    assert cls is None or isinstance(data, cls)
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)


T_co = TypeVar('T_co', covariant=True)


def loadValue(path: str, cls: Type[T_co] = None) -> T_co:
    """
    Loads data from the given path.
    :param path:  Path to load the data from.
    :param cls:   Class type for validating the data.
    :return:  Loaded data
    """
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    assert cls is None or isinstance(data, cls)
    return data
