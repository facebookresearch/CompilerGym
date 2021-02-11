# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict, List

from compiler_gym.spaces.reward import Reward
from compiler_gym.views.observation import ObservationView


class RewardView(object):
    """A view into a set of reward spaces.

    Example usage:

    >>> env = gym.make("llvm-v0")
    >>> env.reset()
    >>> env.reward.spaces["codesize"].range
    (-np.inf, 0)
    >>> env.reward["codesize"]
    -1243

    :ivar spaces: Specifications of available reward spaces.
    :vartype spaces: Dict[str, Reward]
    """

    def __init__(
        self,
        spaces: List[Reward],
        observation_view: ObservationView,
    ):
        self.spaces: Dict[str, Reward] = {s.id: s for s in spaces}
        self.previous_action = None
        self._observation_view = observation_view

    def __getitem__(self, reward_space: str) -> float:
        """Request an observation from the given space.

        :param reward_space: The reward space to query.
        :return: A reward.
        :raises KeyError: If the requested reward space does not exist.
        """
        # TODO(cummins): Since reward is a function from (state, action) -> r
        # it would be better to make the list of reward to evaluate an argument
        # to env.step() rather than using this lazy view.
        if not self.spaces:
            raise ValueError("No reward spaces")
        space = self.spaces[reward_space]
        observations = [self._observation_view[obs] for obs in space.observation_spaces]
        return space.update(self.previous_action, observations, self._observation_view)

    def reset(self) -> None:
        """Reset the rewards space view. This is called on
        :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>`."""
        self.previous_action = None
        for space in self.spaces.values():
            space.reset()

    def add_space(self, space: Reward) -> None:
        """Register a new reward space.

        :param space: The reward space to be added.
        """
        if space.id in self.spaces:
            warnings.warn(f"Replacing existing reward space '{space.id}'")
        self.spaces[space.id] = space
