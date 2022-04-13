# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from numbers import Integral

import numpy as np

from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.sequence import Sequence


class Permutation(Sequence):
    """The space of permutations of all numbers in the range `scalar_range`."""

    def __init__(self, name: str, scalar_range: Scalar):
        """Constructor.

        :param name: The name of the permutation space.
        :param scalar_range: Range of numbers in the permutation.
            For example the scalar range [1, 3] would define permutations like
            [1, 2, 3] or [2, 1, 3], etc.

        :raises TypeError: If `scalar_range.dtype` is not an integral type.
        """
        if not issubclass(np.dtype(scalar_range.dtype).type, Integral):
            raise TypeError("Permutation space can have integral scalar range only.")
        sz = scalar_range.max - scalar_range.min + 1
        super().__init__(
            name=name,
            size_range=(sz, sz),
            dtype=scalar_range.dtype,
            scalar_range=scalar_range,
        )

    def sample(self):
        return (
            np.random.choice(self.size_range[0], size=self.size_range[1], replace=False)
            + self.scalar_range.min
        )

    def __eq__(self, other) -> bool:
        return isinstance(self, other.__class__) and super().__eq__(other)
