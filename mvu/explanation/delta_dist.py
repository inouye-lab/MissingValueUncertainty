from numbers import Number
from typing import Union

import torch
from overrides import override
from torch import Tensor
from torch.distributions import Distribution, constraints


class DeltaDistribution(Distribution):
    """
    This class implements a "Distribution" which has a fixed mean and 0 variance. It allows generalizing our code to
    support any Torch distribution without needing to special case setups without uncertainty.
    """
    arg_constraints = {"loc": constraints.real}
    support = constraints.real
    has_rsample = True
    loc: Tensor
    zeros: Tensor

    def __init__(self, loc: Union[Tensor, Number], validate_args=None):
        """
        Creates a new instance of this distribution
        :param loc:            Mean value for the distribution
        :param validate_args:  If false, disables argument validation.
        """
        if isinstance(loc, Number):
            self.loc = torch.tensor(loc)
            batch_shape = torch.Size()
        else:
            self.loc = loc
            batch_shape = loc.size()
        self.zeros = torch.zeros_like(self.loc)
        super().__init__(batch_shape, validate_args=validate_args)

    @override
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DeltaDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.zeros = self.zeros.expand(batch_shape)
        super(DeltaDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    @override
    def mean(self) -> Tensor:
        return self.loc

    @property
    @override
    def mode(self) -> Tensor:
        return self.loc

    @property
    @override
    def stddev(self) -> Tensor:
        return self.zeros

    @property
    @override
    def variance(self) -> Tensor:
        return self.zeros

    @override
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.loc.expand(sample_shape)

    @override
    def log_prob(self, value: Tensor) -> Tensor:
        return torch.where(torch.eq(value, self.loc), torch.tensor(0), -torch.inf)

    @override
    def cdf(self, value: Tensor) -> Tensor:
        return torch.where(torch.ge(value, self.loc), torch.tensor(1), torch.tensor(0))

    @override
    def entropy(self) -> torch.Tensor:
        return self.zeros
