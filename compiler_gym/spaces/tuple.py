# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from gym.spaces import Space
from gym.spaces import Tuple as GymTuple


class Tuple(GymTuple):
    """A tuple (i.e., product) of simpler spaces.

    Wraps the underlying :code:`gym.spaces.Tuple` space with a name attribute.
    """

    def __init__(self, spaces: List[Space], name: str):
        """Constructor.

        :param spaces: The composite spaces.

        :param name: The name of the space.
        """
        super().__init__(spaces)
        self.name = name

    def __eq__(self, other) -> bool:
        return (
            isinstance(self, other.__class__)
            and self.name == other.name
            and super().__eq__(other)
        )
