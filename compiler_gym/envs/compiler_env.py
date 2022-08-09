# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import gym
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import (
    ActionType,
    ObservationType,
    OptionalArgumentValue,
    StepType,
)
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView


class CompilerEnv(gym.Env, ABC):
    """An OpenAI gym environment for compiler optimizations.

    The easiest way to create a CompilerGym environment is to call
    :code:`gym.make()` on one of the registered environments:

        >>> env = gym.make("llvm-v0")

    See :code:`compiler_gym.COMPILER_GYM_ENVS` for a list of registered
    environment names.

    Alternatively, an environment can be constructed directly, such as by
    connecting to a running compiler service at :code:`localhost:8080` (see
    :doc:`this document </compiler_gym/service>` for more details):

        >>> env = ClientServiceCompilerEnv(
        ...     service="localhost:8080",
        ...     observation_space="features",
        ...     reward_space="runtime",
        ...     rewards=[env_reward_spaces],
        ... )

    Once constructed, an environment can be used in exactly the same way as a
    regular :code:`gym.Env`, e.g.

        >>> observation = env.reset()
        >>> cumulative_reward = 0
        >>> for i in range(100):
        >>>     action = env.action_space.sample()
        >>>     observation, reward, done, info = env.step(action)
        >>>     cumulative_reward += reward
        >>>     if done:
        >>>         break
        >>> print(f"Reward after {i} steps: {cumulative_reward}")
        Reward after 100 steps: -0.32123
    """

    @abstractmethod
    def __init__(self):
        """Construct an environment.

        Do not construct an environment directly. Use :code:`gym.make()` on one
        of the registered environments:

        >>> with gym.make("llvm-v0") as env:
        ...     pass  # Use environment
        """
        raise NotImplementedError("abstract class")

    @abstractmethod
    def close(self):
        """Close the environment.

        Once closed, :func:`reset` must be called before the environment is used
        again.

        .. note::

            You must make sure to call :code:`env.close()` on a CompilerGym
            environment when you are done with it. This is needed to perform
            manual tidying up of temporary files and processes. See :ref:`the
            FAQ <faq:Do I need to call env.close()?>` for more details.
        """
        raise NotImplementedError("abstract method")

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

    @property
    @abstractmethod
    def reward_space_spec(self) -> Optional[Reward]:
        raise NotImplementedError("abstract method")

    @reward_space_spec.setter
    @abstractmethod
    def reward_space_spec(self, val: Optional[Reward]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def benchmark(self) -> Benchmark:
        """Get or set the benchmark to use.

        :getter: Get :class:`Benchmark <compiler_gym.datasets.Benchmark>` that
            is currently in use.

        :setter: Set the benchmark to use. Either a :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instance, or the URI of a
            benchmark as in :meth:`env.datasets.benchmark_uris()
            <compiler_gym.datasets.Datasets.benchmark_uris>`.

        .. note::

            Setting a new benchmark has no effect until
            :func:`env.reset() <compiler_gym.envs.CompilerEnv.reset>` is called.
        """
        raise NotImplementedError("abstract method")

    @benchmark.setter
    @abstractmethod
    def benchmark(self, benchmark: Optional[Union[str, Benchmark, BenchmarkUri]]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def datasets(self) -> Iterable[Dataset]:
        raise NotImplementedError("abstract method")

    @datasets.setter
    @abstractmethod
    def datasets(self, datasets: Iterable[Dataset]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def episode_walltime(self) -> float:
        """Return the amount of time in seconds since the last call to
        :meth:`reset() <compiler_gym.envs.CompilerEnv.reset>`.
        """
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def in_episode(self) -> bool:
        """Whether the service is ready for :func:`step` to be called,
        i.e. :func:`reset` has been called and :func:`close` has not.

        :return: :code:`True` if in an episode, else :code:`False`.
        """
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def episode_reward(self) -> Optional[float]:
        """If :func:`CompilerEnv.reward_space
        <compiler_gym.envs.CompilerGym.reward_space>` is set, this value is the
        sum of all rewards for the current episode.
        """
        raise NotImplementedError("abstract method")

    @episode_reward.setter
    @abstractmethod
    def episode_reward(self, episode_reward: Optional[float]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def actions(self) -> List[ActionType]:
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def version(self) -> str:
        """The version string of the compiler service."""
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def compiler_version(self) -> str:
        """The version string of the underlying compiler that this service supports."""
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def state(self) -> CompilerEnvState:
        """The tuple representation of the current environment state."""
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """The current action space.

        :getter: Get the current action space.
        :setter: Set the action space to use. Must be an entry in
            :code:`action_spaces`. If :code:`None`, the default action space is
            selected.
        """
        raise NotImplementedError("abstract method")

    @action_space.setter
    @abstractmethod
    def action_space(self, action_space: Optional[str]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def action_spaces(self) -> List[str]:
        """A list of supported action space names."""
        raise NotImplementedError("abstract method")

    @action_spaces.setter
    @abstractmethod
    def action_spaces(self, action_spaces: List[str]):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def reward_space(self) -> Optional[Reward]:
        """The default reward space that is used to return a reward value from
        :func:`~step()`.

        :getter: Returns a :class:`Reward <compiler_gym.spaces.Reward>`,
            or :code:`None` if not set.
        :setter: Set the default reward space.
        """
        raise NotImplementedError("abstract method")

    @reward_space.setter
    @abstractmethod
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def observation_space(self) -> Optional[Space]:
        """The observation space that is used to return an observation value in
        :func:`~step()`.

        :getter: Returns the underlying observation space, or :code:`None` if
            not set.
        :setter: Set the default observation space.
        """
        raise NotImplementedError("abstract method")

    @observation_space.setter
    @abstractmethod
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def observation(self) -> ObservationView:
        """A view of the available observation spaces that permits
        on-demand computation of observations.
        """
        raise NotImplementedError("abstract method")

    @observation.setter
    @abstractmethod
    def observation(self, observation: ObservationView) -> None:
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def reward_range(self) -> Tuple[float, float]:
        """A tuple indicating the range of reward values.

        Default range is (-inf, +inf).
        """
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def reward(self) -> RewardView:
        """A view of the available reward spaces that permits on-demand
        computation of rewards.
        """
        raise NotImplementedError("abstract method")

    @reward.setter
    @abstractmethod
    def reward(self, reward: RewardView) -> None:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def fork(self) -> "CompilerEnv":
        """Fork a new environment with exactly the same state.

        This creates a duplicate environment instance with the current state.
        The new environment is entirely independently of the source environment.
        The user must call :meth:`close() <compiler_gym.envs.CompilerEnv.close>`
        on the original and new environments.

        If not already in an episode, :meth:`reset()
        <compiler_gym.envs.CompilerEnv.reset>` is called.

        Example usage:

            >>> env = gym.make("llvm-v0")
            >>> env.reset()
            # ... use env
            >>> new_env = env.fork()
            >>> new_env.state == env.state
            True
            >>> new_env.step(1) == env.step(1)
            True

        .. note::

            The client/service implementation of CompilerGym means that the
            forked and base environments share a common backend resource. This
            means that if either of them crash, such as due to a compiler
            assertion, both environments must be reset.

        :return: A new environment instance.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset(  # pylint: disable=arguments-differ
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        observation_space: Union[
            OptionalArgumentValue, str, ObservationSpaceSpec
        ] = OptionalArgumentValue.UNCHANGED,
        reward_space: Union[
            OptionalArgumentValue, str, Reward
        ] = OptionalArgumentValue.UNCHANGED,
        timeout: float = 300,
    ) -> Optional[ObservationType]:
        """Reset the environment state.

        This method must be called before :func:`step()`.

        :param benchmark: The name of the benchmark to use. If provided, it
            overrides any value that was set during :func:`__init__`, and
            becomes subsequent calls to :code:`reset()` will use this benchmark.
            If no benchmark is provided, and no benchmark was provided to
            :func:`__init___`, the service will randomly select a benchmark to
            use.

        :param action_space: The name of the action space to use. If provided,
            it overrides any value that set during :func:`__init__`, and
            subsequent calls to :code:`reset()` will use this action space. If
            no action space is provided, the default action space is used.

        :param observation_space: Compute and return observations at each
            :func:`step()` from this space. Accepts a string name or an
            :class:`ObservationSpaceSpec
            <compiler_gym.views.ObservationSpaceSpec>`. If :code:`None`,
            :func:`step()` returns :code:`None` for the observation value. If
            :code:`OptionalArgumentValue.UNCHANGED` (the default value), the
            observation space remains unchanged from the previous episode. For
            available spaces, see :class:`env.observation.spaces
            <compiler_gym.views.ObservationView>`.

        :param reward_space: Compute and return reward at each :func:`step()`
            from this space. Accepts a string name or a :class:`Reward
            <compiler_gym.spaces.Reward>`. If :code:`None`, :func:`step()`
            returns :code:`None` for the reward value.  If
            :code:`OptionalArgumentValue.UNCHANGED` (the default value), the
            observation space remains unchanged from the previous episode. For
            available spaces, see :class:`env.reward.spaces
            <compiler_gym.views.RewardView>`.

        :param timeout: The maximum number of seconds to wait for reset to
            succeed.

        :return: The initial observation.

        :raises BenchmarkInitError: If the benchmark is invalid. This can happen
            if the benchmark contains code that the compiler does not support,
            or because of some internal error within the compiler. In this case,
            another benchmark must be used.

        :raises TypeError: If no benchmark has been set, and the environment
            does not have a default benchmark to select from.
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
        timeout: float = 300,
    ) -> StepType:
        """Take a step.

        :param action: An action.

        :param observation_spaces: A list of observation spaces to compute
            observations from. If provided, this changes the :code:`observation`
            element of the return tuple to be a list of observations from the
            requested spaces. The default :code:`env.observation_space` is not
            returned.

        :param reward_spaces: A list of reward spaces to compute rewards from.
            If provided, this changes the :code:`reward` element of the return
            tuple to be a list of rewards from the requested spaces. The default
            :code:`env.reward_space` is not returned.

        :param timeout: The maximum number of seconds to wait for the step to
            succeed. Accepts a float value. The default is 300 seconds.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: float = 300,
    ):
        """Take a sequence of steps and return the final observation and reward.

        :param action: A sequence of actions to apply in order.

        :param observation_spaces: A list of observation spaces to compute
            observations from. If provided, this changes the :code:`observation`
            element of the return tuple to be a list of observations from the
            requested spaces. The default :code:`env.observation_space` is not
            returned.

        :param reward_spaces: A list of reward spaces to compute rewards from.
            If provided, this changes the :code:`reward` element of the return
            tuple to be a list of rewards from the requested spaces. The default
            :code:`env.reward_space` is not returned.

        :param timeout: The maximum number of seconds to wait for the steps to
            succeed. Accepts a float value. The default is 300 seconds.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        """Render the environment.

        :param mode: The render mode to use.
        :raises TypeError: If a default observation space is not set, or if the
            requested render mode does not exist.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def commandline(self) -> str:
        """Interface for :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
        subclasses to provide an equivalent commandline invocation to the
        current environment state.

        See also :meth:`commandline_to_actions()
        <compiler_gym.envs.CompilerEnv.commandline_to_actions>`.

        :return: A string commandline invocation.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def commandline_to_actions(self, commandline: str) -> List[ActionType]:
        """Interface for :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
        subclasses to convert from a commandline invocation to a sequence of
        actions.

        See also :meth:`commandline()
        <compiler_gym.envs.CompilerEnv.commandline>`.

        :return: A list of actions.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def apply(self, state: CompilerEnvState) -> None:  # noqa
        """Replay this state on the given environment.

        :param state: A :class:`CompilerEnvState <compiler_gym.CompilerEnvState>`
            instance.

        :raises ValueError: If this state cannot be applied.
        """
        raise NotImplementedError("abstract method")

    @abstractmethod
    def validate(self, state: Optional[CompilerEnvState] = None) -> ValidationResult:
        """Validate an environment's state.

        :param state: A state to environment. If not provided, the current state
            is validated.

        :returns: A :class:`ValidationResult <compiler_gym.ValidationResult>`.
        """
        raise NotImplementedError("abstract method")
