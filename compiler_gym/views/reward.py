# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, List

from compiler_gym.service.proto import Reward, RewardRequest, RewardSpace
from compiler_gym.views.reward_space_spec import RewardSpaceSpec


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
    :vartype spaces: Dict[str, RewardSpaceSpec]
    """

    def __init__(
        self,
        get_reward: Callable[[RewardRequest], Reward],
        spaces: List[RewardSpace],
    ):
        self._get_reward = get_reward
        self.session_id = -1

        if not spaces:
            raise ValueError("No reward spaces")

        self.spaces = {s.name: RewardSpaceSpec(i, s) for i, s in enumerate(spaces)}

    def __getitem__(self, reward_space: str) -> float:
        """Request an observation from the given space.

        :param reward_space: The reward space to query.
        :return: A reward.
        :raises KeyError: If the requested reward space does not exist.
        """
        request = RewardRequest(
            session_id=self.session_id, reward_space=self.spaces[reward_space].index
        )
        return self._get_reward(request).reward
