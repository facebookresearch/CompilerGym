# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from collections.abc import Collection
from typing import Optional, Tuple

import numpy as np
from gym.spaces import Space


class SpaceSequence(Space):
    name: str
    space: Space
    size_range: Tuple[int, Optional[int]]

    def __init__(
        self, name: str, space: Space, size_range: Tuple[int, Optional[int]] = (0, None)
    ):
        self.name = name
        self.space = space
        self.size_range = size_range

    def contains(self, x):
        if not isinstance(x, Collection):
            return False

        lower_bound = self.size_range[0]
        upper_bound = float("inf") if self.size_range[1] is None else self.size_range[1]
        if not (lower_bound <= len(x) <= upper_bound):
            return False

        for element in x:
            if not self.space.contains(element):
                return False
        return True

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, other.__class__)
            and self.name == other.name
            and Counter(self.size_range) == Counter(other.size_range)
            and self.space == other.space
        )

    def sample(self):
        return [
            self.space.sample()
            for _ in range(
                np.random.randint(
                    low=self.size_range[0],
                    high=None if self.size_range[1] is None else self.size_range[1] + 1,
                )
            )
        ]
