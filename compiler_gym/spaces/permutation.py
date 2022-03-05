# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.sequence import Sequence


class Permutation(Sequence):
    def __init__(self, name: str, scalar_range: Scalar):
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
        return (
            isinstance(self, other.__class__)
            and self.name == other.name
            and super().__eq__(other)
        )
