from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal

from .dataset import Dataset
from .regressor import Regressor


def estimateResidual(regressor: Regressor, features: Union[Tensor, Dataset], targets: Tensor = None) -> Tensor:
    """
    Estimates the residual uncertainty for the given regressor and features tensor
    :param regressor: Regressor instance
    :param features:  Feature tensor of size `(samples, features)` with no missingness
    :param targets:   Targets for each sample, size `(samples,)`
    :return: Residual uncertainty for the whole model as a tensor of size 1
    """
    # allow passing a dataset as the second parameter instead of splitting it
    if targets is None:
        targets = features.targets
        features = features.features

    # make prediction, then return mean squared error
    prediction = regressor.predict(features)
    return torch.mean((targets - prediction) ** 2)
