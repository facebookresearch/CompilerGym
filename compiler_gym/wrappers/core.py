# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Optional, Union

import gym

from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ObservationType, StepType
from compiler_gym.views import ObservationSpaceSpec


class CompilerEnvWrapper(gym.Wrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow a modular transformation.

    This class is the base class for all wrappers. This class must be used
    rather than :code:`gym.Wrapper` to support the CompilerGym API extensions
    such as the :code:`fork()` method.
    """

    def __init__(self, env: CompilerEnv):
        """Constructor.

        :param env: The environment to wrap.

        :raises TypeError: If :code:`env` is not a :class:`CompilerEnv
            <compiler_gym.envs.CompilerEnv>`.
        """
        # No call to gym.Wrapper superclass constructor here because we need to
        # avoid setting the observation_space member variable, which in the
        # CompilerEnv class is a property with a custom setter. Instead we set
        # the observation_space_spec directly.
        self.env = env
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def step(self, action, observations=None, rewards=None):
        return self.env.step(action, observations=observations, rewards=rewards)

    def reset(self, *args, **kwargs) -> ObservationType:
        return self.env.reset(*args, **kwargs)

    def fork(self) -> CompilerEnv:
        return type(self)(env=self.env.fork())

    @property
    def observation_space(self):
        if self.env.observation_space_spec:
            return self.env.observation_space_spec.space

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        self.env.observation_space = observation_space

    @property
    def observation_space_spec(self):
        return self.env.observation_space_spec

    @observation_space_spec.setter
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ) -> None:
        self.env.observation_space_spec = observation_space_spec

    @property
    def reward_space(self) -> Optional[Reward]:
        return self.env.reward_space

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        self.env.reward_space = reward_space


class ActionWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an action space transformation.
    """

    def step(
        self, action: Union[int, Iterable[int]], observations=None, rewards=None
    ) -> StepType:
        return self.env.step(
            self.action(action), observations=observations, rewards=rewards
        )

    def action(self, action):
        """Translate the action to the new space."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Translate an action from the new space to the wrapped space."""
        raise NotImplementedError


class ObservationWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an observation space transformation.
    """

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        """Translate an observation to the new space."""
        raise NotImplementedError


class RewardWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an reward space transformation.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.env.step(*args, **kwargs)
        # Undo the episode_reward update and reapply it once we have transformed
        # the reward.
        #
        # TODO(cummins): Refactor step() so that we don't have to do this
        # recalculation of episode_reward, as this is prone to errors if, say,
        # the base reward returns NaN or an invalid type.
        if reward is not None and self.episode_reward is not None:
            self.unwrapped.episode_reward -= reward
            reward = self.reward(reward)
            self.unwrapped.episode_reward += reward
        return observation, reward, done, info

    def reward(self, reward):
        """Translate a reward to the new space."""
        raise NotImplementedError
