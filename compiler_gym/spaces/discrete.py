# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from gym.spaces import Discrete as GymDiscrete


class Discrete(GymDiscrete):
    """A discrete space in :math:`{ 0, 1, \\dots, n-1 }`.

    Wraps the underlying :code:`gym.spaces.Discrete` space with a name attribute.
    """

    def __init__(self, n: int, name: str):
        """Constructor.

        :param n: The upper bound.

        :param name: The name of the space.
        """
        super().__init__(n)
        self.name = name
