# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Reward spaces for use in the GCC environments."""
from compiler_gym.spaces import Reward
from compiler_gym.views.observation import ObservationView


class AsmSizeReward(Reward):
    """Reward for the size in bytes of the assembly code"""

    def __init__(self):
        super().__init__(
            name="asm_size",
            observation_spaces=["asm_size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str, observation_view: ObservationView):
        super().reset(benchmark, observation_view)
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward


class ObjSizeReward(Reward):
    """Reward for the size in bytes of the object code"""

    def __init__(self):
        super().__init__(
            name="obj_size",
            observation_spaces=["obj_size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str, observation_view: ObservationView):
        super().reset(benchmark, observation_view)
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward
