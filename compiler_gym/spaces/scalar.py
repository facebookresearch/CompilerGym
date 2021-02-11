# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Optional

from gym.spaces import Space


class Scalar(Space):
    """A scalar value."""

    __slots__ = ["min", "max", "dtype"]

    def __init__(
        self, min: Optional[float] = None, max: Optional[float] = None, dtype=float
    ):
        """Constructor.

        :param min: The lower bound for a value in this space. If None, there is
            no lower bound.
        :param max: The upper bound for a value in this space. If None, there is
            no upper bound.
        :param dtype: The type of this scalar.
        """
        self.min = min
        self.max = max
        self.dtype = dtype

    def sample(self):
        min = 0 if self.min is None else self.min
        max = 1 if self.max is None else self.max
        return self.dtype(random.uniform(min, max))

    def contains(self, x):
        if not isinstance(x, self.dtype):
            return False
        min = -float("inf") if self.min is None else self.min
        max = float("inf") if self.max is None else self.max
        return min <= x <= max

    def __repr__(self):
        if self.min is None and self.max is None:
            return self.dtype.__name__
        lower_bound = "-inf" if self.min is None else self.min
        upper_bound = "inf" if self.max is None else self.max
        return f"{self.dtype.__name__}<{lower_bound},{upper_bound}>"

    def __eq__(self, rhs):
        """Equality test."""
        if not isinstance(rhs, Scalar):
            return False
        return self.min == rhs.min and self.max == rhs.max and self.dtype == rhs.dtype
