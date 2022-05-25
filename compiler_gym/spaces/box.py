# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Optional

import numpy as np
from gym.spaces import Box as GymBox


class Box(GymBox):
    """A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    Wraps the underlying :code:`gym.spaces.Box` with a name attribute.
    """

    def __init__(
        self,
        low: float,
        high: float,
        name: str,
        shape: Optional[Iterable[int]] = None,
        dtype=np.float32,
    ):
        """Constructor.

        :param low: The lower bound, inclusive.

        :param high: The upper bound, inclusive.

        :param name: The name of the space.

        :param shape: The shape of the space.

        :param dtype: The dtype of the space.
        """
        super().__init__(low=low, high=high, shape=shape, dtype=dtype)
        self.name = name

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, other.__class__)
            and self.name == other.name
            and super().__eq__(other)
        )
