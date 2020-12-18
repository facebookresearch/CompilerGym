# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Sequence

from compiler_gym.service import scalar_range2tuple
from compiler_gym.service.proto import Reward, RewardRequest, RewardSpace


class RewardView(object):
    """A view into a set of reward spaces.

    Example usage:

    >>> env = gym.make("llvm-v0")
    >>> env.reset()
    >>> env.reward.ranges.keys()
    ["codesize"]
    >>> env.reward.ranges["codesize"]
    (-np.inf, 0)
    >>> env.reward["codesize"]
    -1243
    """

    def __init__(
        self,
        get_reward: Callable[[RewardRequest], Reward],
        spaces: Sequence[RewardSpace],
    ):
        self._get_reward = get_reward
        self.session_id = -1

        if not spaces:
            raise ValueError("No reward spaces")

        self.indices = {s.name: i for i, s in enumerate(spaces)}
        self.ranges = {s.name: scalar_range2tuple(s.range) for s in spaces}

    def __getitem__(self, reward_space: str) -> float:
        """Request an observation from the given space.

        :param reward_space: The reward space to query.
        :return: A reward.
        :raises KeyError: If the requested reward space does not exist.
        """
        request = RewardRequest(
            session_id=self.session_id, reward_space=self.indices[reward_space]
        )
        return self._get_reward(request).reward

    # TODO(cummins): Copy the register_derived_space() functionality from
    # ObservationView to allow derived reward spaces, possible creating a shared
    # superclass.
