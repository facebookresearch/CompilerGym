# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.service import scalar_range2tuple
from compiler_gym.service.proto import RewardSpace


class RewardSpaceSpec(object):
    """Specification of a reward space.

    :ivar id: The name of the reward space.
    :vartype id: str

    :ivar index: The index into the list of reward spaces that the service
        supports.
    :vartype index: int

    :ivar range: The lower and upper bounds of the reward.
    :vartype range: Tuple[float, float]

    :ivar success_threshold: The cumulative reward threshold before an episode is
        considered successful. For example, episodes where reward is scaled to
        an existing heuristic can be considered "successful" when the reward
        exceeds the existing heuristic.
    :vartype success_threshold: Optional[float]

    :ivar deterministic: Whether the reward signal is deterministic.
        Examples of non-deterministic rewards are those with measurement noise
        such as program runtime.
    :vartype deterministic: bool

    :ivar platform_dependent: Whether the reward values depend on the execution
        environment of the service.
    :vartype platform_dependent: bool

    :ivar default_value: A default reward value. This value will be returned by
        :func:`CompilerEnv.step() <compiler_gym.envs.CompilerEnv.step>` if
        :func:`CompilerEnv.reward_space <compiler_gym.envs.CompilerEnv.reward_space>`
        is set and the service terminates.
    :vartype default_value: float

    :ivar default_negates_returns: If true, the default value will be offset by
        the sum of all rewards for the current episode. For example, given a
        default reward value of -10.0 and an episode with prior rewards
        [0.1, 0.3, -0.15], the default value is: -10.0 - sum(0.1, 0.3, -0.15).
    :vartype default_negates_returns: bool
    """

    def __init__(self, index: int, spec: RewardSpace):
        self.id = spec.name
        self.index = index
        self.range = scalar_range2tuple(spec.range)
        self.success_threshold = (
            spec.success_threshold if spec.has_success_threshold else None
        )
        self.deterministic = spec.deterministic
        self.platform_dependent = spec.platform_dependent
        self.default_value = spec.default_value
        self.default_negates_returns = spec.default_negates_returns

    def __repr__(self) -> str:
        return f"RewardSpaceSpec({self.id})"

    def reward_on_error(self, episode_reward: float) -> float:
        """Return the reward value for an error condition.

        This method should be used to produce the reward value that should be
        used if the compiler service cannot be reached, e.g. because it has
        crashed or the connection has dropped.

        :param episode_reward: The current cumulative reward of an episode.
        :return: A reward.
        """
        if self.default_negates_returns:
            return self.default_value - episode_reward
        else:
            return self.default_value
