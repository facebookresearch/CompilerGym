# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from collections.abc import Iterable as IterableType
from typing import Iterable, Optional, Union

import gym

from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType
from compiler_gym.views import ObservationSpaceSpec


class CompilerEnvWrapper(gym.Wrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow a modular transformation.

    This class is the base class for all wrappers. This class must be used
    rather than :code:`gym.Wrapper` to support the CompilerGym API extensions
    such as the :code:`fork()` method.
    """

    def __init__(self, env: CompilerEnv):  # pylint: disable=super-init-not-called
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

    def raw_step(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Iterable[ObservationSpaceSpec],
        reward_spaces: Iterable[Reward],
    ):
        return self.env.raw_step(
            actions, observation_spaces=observation_spaces, reward_spaces=reward_spaces
        )

    def reset(self, *args, **kwargs) -> ObservationType:
        return self.env.reset(*args, **kwargs)

    def fork(self) -> CompilerEnv:
        return type(self)(env=self.env.fork())

    # NOTE(cummins): This step() method is provided only because
    # CompilerEnv.step accepts additional arguments over gym.Env.step. Users who
    # wish to modify the behavior of CompilerEnv.step should overload
    # raw_step().
    def step(  # pylint: disable=arguments-differ
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        if isinstance(action, IterableType):
            warnings.warn(
                "Argument `action` of CompilerEnv.step no longer accepts a list "
                " of actions. Please use CompilerEnv.multistep instead",
                category=DeprecationWarning,
            )
            return self.multistep(
                action,
                observation_spaces=observation_spaces,
                reward_spaces=reward_spaces,
                observations=observations,
                rewards=rewards,
            )
        if observations is not None:
            warnings.warn(
                "Argument `observations` of CompilerEnv.multistep has been "
                "renamed `observation_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            observation_spaces = observations
        if rewards is not None:
            warnings.warn(
                "Argument `rewards` of CompilerEnv.multistep has been renamed "
                "`reward_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            reward_spaces = rewards
        return self.env._multistep(
            raw_step=self.raw_step,
            actions=[action],
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
        )

    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        if observations is not None:
            warnings.warn(
                "Argument `observations` of CompilerEnv.multistep has been "
                "renamed `observation_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            observation_spaces = observations
        if rewards is not None:
            warnings.warn(
                "Argument `rewards` of CompilerEnv.multistep has been renamed "
                "`reward_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            reward_spaces = rewards
        return self.env._multistep(  # pylint: disable=protected-access
            raw_step=self.raw_step,
            actions=actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
        )

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

    def raw_step(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Iterable[ObservationSpaceSpec],
        reward_spaces: Iterable[Reward],
    ):
        return self.env.raw_step(
            [self.action(a) for a in actions],
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
        )

    def action(self, action: ActionType) -> ActionType:
        """Translate the action to the new space."""
        raise NotImplementedError

    def reverse_action(self, action: ActionType) -> ActionType:
        """Translate an action from the new space to the wrapped space."""
        raise NotImplementedError


class ObservationWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an observation space transformation.
    """

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    def raw_step(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Iterable[ObservationSpaceSpec],
        reward_spaces: Iterable[Reward],
    ):
        observation, reward, done, info = self.env.raw_step(
            actions, observation_spaces=observation_spaces, reward_spaces=reward_spaces
        )

        # Only apply observation transformation if we are using the default
        # observation space.
        if observation_spaces == [self.observation_space_spec]:
            observation = [self.observation(observation)]

        return observation, reward, done, info

    def observation(self, observation):
        """Translate an observation to the new space."""
        raise NotImplementedError


class RewardWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an reward space transformation.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def raw_step(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Iterable[ObservationSpaceSpec],
        reward_spaces: Iterable[Reward],
    ):
        observation, reward, done, info = self.env.step(
            actions, observation_spaces=observation_spaces, reward_spaces=reward_spaces
        )

        # Only apply rewards transformation if we are using the default
        # reward space.
        if reward_spaces == [self.reward_space]:
            reward = [self.reward(reward)]

        return observation, reward, done, info

    def reward(self, reward):
        """Translate a reward to the new space."""
        raise NotImplementedError
