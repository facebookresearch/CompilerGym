# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional, Tuple

import numpy as np

from compiler_gym.spaces.scalar import Scalar
from compiler_gym.util.gym_type_hints import ObservationType, RewardType


class Reward(Scalar):
    """An extension of the :class:`Scalar <compiler_gym.spaces.Scalar>` space
    that is used for computing a reward signal.

    A :code:`Reward` is a scalar value used to determine the reward for a
    particular action. An instance of :code:`Reward` is used to represent the
    reward function for a particular episode. For every
    :meth:`env.step() <compiler_gym.envs.CompilerEnv.step>` of the environment,
    the :meth:`reward.update() <compiler_gym.spaces.Reward.update>` method is
    called to produce a new incremental reward.

    Environments provide implementations of :code:`Reward` that compute reward
    signals based on observation values computed by the backend service.
    """

    __slots__ = [
        "id",
        "observation_spaces",
        "default_value",
        "default_negates_returns",
        "success_threshold",
        "deterministic",
        "platform_dependent",
    ]

    def __init__(
        self,
        id: str,
        observation_spaces: Optional[List[str]] = None,
        default_value: RewardType = 0,
        min: Optional[RewardType] = None,
        max: Optional[RewardType] = None,
        default_negates_returns: bool = False,
        success_threshold: Optional[RewardType] = None,
        deterministic: bool = False,
        platform_dependent: bool = True,
    ):
        """Constructor.

        :param id: The ID of the reward space. This is a unique name used to
            represent the reward.
        :param observation_spaces: A list of observation space IDs
            (:class:`space.id <compiler_gym.views.ObservationSpaceSpec>` values)
            that are used to compute the reward. May be an empty list if no
            observations are requested. Requested observations will be provided
            to the :code:`observations` argument of
            :meth:`reward.update() <compiler_gym.spaces.Reward.update>`.
        :param default_value: A default reward. This value will be returned by
            :meth:`env.step() <compiler_gym.envs.CompilerEnv.step>` if
            the service terminates.
        :param min: The lower bound of the reward.
        :param max: The upper bound of the reward.
        :param default_negates_returns: If true, the default value will be
            offset by the sum of all rewards for the current episode. For
            example, given a default reward value of *-10.0* and an episode with
            prior rewards *[0.1, 0.3, -0.15]*, the default value is:
            *-10.0 - sum(0.1, 0.3, -0.15)*.
        :param success_threshold: The cumulative reward threshold before an
            episode is considered successful. For example, episodes where reward
            is scaled to an existing heuristic can be considered “successful”
            when the reward exceeds the existing heuristic.
        :param deterministic: Whether the reward space is deterministic.
        :param platform_dependent: Whether the reward values depend on the
            execution environment of the service.
        """
        super().__init__(
            min=-np.inf if min is None else min,
            max=np.inf if max is None else max,
            dtype=np.float64,
        )
        self.id = id
        self.observation_spaces = observation_spaces or []
        self.default_value: RewardType = default_value
        self.default_negates_returns: bool = default_negates_returns
        self.success_threshold = success_threshold
        self.deterministic = deterministic
        self.platform_dependent = platform_dependent

    def reset(self, benchmark: str) -> None:
        """Reset the rewards space. This is called on
        :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>`.

        :param benchmark: The URI of the benchmark that is used for this
            episode.
        """
        pass

    def update(
        self,
        action: int,
        observations: List[ObservationType],
        observation_view: "compiler_gym.views.ObservationView",  # noqa: F821
    ) -> RewardType:
        """Calculate a reward for the given action.

        :param action: The action performed.
        :param observations: A list of observation values as requested by the
            :code:`observation_spaces` constructor argument.
        :param observation_view: The
            :class:`ObservationView <compiler_gym.views.ObservationView>`
            instance.
        """
        raise NotImplementedError("abstract class")

    def reward_on_error(self, episode_reward: RewardType) -> RewardType:
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

    @property
    def range(self) -> Tuple[RewardType, RewardType]:
        """The lower and upper bounds of the reward."""
        return (self.min, self.max)

    def __repr__(self):
        return self.id


class DefaultRewardFromObservation(Reward):
    def __init__(self, observation_name: str, **kwargs):
        super().__init__(
            observation_spaces=[observation_name], id=observation_name, **kwargs
        )
        self.previous_value: Optional[ObservationType] = None

    def reset(self, benchmark: str) -> None:
        """Called on env.reset(). Reset incremental progress."""
        del benchmark  # unused
        self.previous_value = None

    def update(
        self,
        action: int,
        observations: List[ObservationType],
        observation_view: "compiler_gym.views.ObservationView",  # noqa: F821
    ) -> RewardType:
        """Called on env.step(). Compute and return new reward."""
        del action  # unused
        del observation_view  # unused
        value: RewardType = observations[0]
        if self.previous_value is None:
            self.previous_value = 0
        reward = RewardType(value - self.previous_value)
        self.previous_value = value
        return reward
