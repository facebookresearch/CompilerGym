# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import gym

from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType, StepType
from compiler_gym.views import ObservationSpaceSpec


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def observation_space_spec(self) -> ObservationSpaceSpec:
        raise NotImplementedError("abstract method")

    @observation_space_spec.setter
    @abstractmethod
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def fork(self) -> "Env":
        """Fork a new environment with exactly the same state.

        This creates a duplicate environment instance with the current state.
        The new environment is entirely independently of the source environment.
        The user must call :meth:`close() <compiler_gym.envs.CompilerEnv.close>`
        on the original and new environments.

        If not already in an episode, :meth:`reset()
        <compiler_gym.envs.Env.reset>` is called.

        Example usage:

            >>> env = gym.make("llvm-v0")
            >>> env.reset()
            # ... use env
            >>> new_env = env.fork()
            >>> new_env.state == env.state
            True
            >>> new_env.step(1) == env.step(1)
            True

        :return: A new environment instance.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset(  # pylint: disable=arguments-differ
        self, *args, **kwargs
    ) -> Optional[ObservationType]:
        """Reset the environment state.

        This method must be called before :func:`step()`.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def step(
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ) -> StepType:
        """Take a step.

        :param action: An action.

        :param observation_spaces: A list of observation spaces to compute
            observations from. If provided, this changes the :code:`observation`
            element of the return tuple to be a list of observations from the
            requested spaces. The default :code:`env.observation_space` is not
            returned.

        :param reward_spaces: A list of reward spaces to compute rewards from. If
            provided, this changes the :code:`reward` element of the return
            tuple to be a list of rewards from the requested spaces. The default
            :code:`env.reward_space` is not returned.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set.
        """
        raise NotImplementedError("abstract method")

    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        """Take a sequence of steps and return the final observation and reward.

        :param action: A sequence of actions to apply in order.

        :param observation_spaces: A list of observation spaces to compute
            observations from. If provided, this changes the :code:`observation`
            element of the return tuple to be a list of observations from the
            requested spaces. The default :code:`env.observation_space` is not
            returned.

        :param reward_spaces: A list of reward spaces to compute rewards from. If
            provided, this changes the :code:`reward` element of the return
            tuple to be a list of rewards from the requested spaces. The default
            :code:`env.reward_space` is not returned.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set.
        """
        raise NotImplementedError("abstract method")
