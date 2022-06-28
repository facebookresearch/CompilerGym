# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable as IterableType
from typing import Any, Iterable, List, Optional, Tuple, Union

from gym import Wrapper
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark, BenchmarkUri, Dataset
from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView


class CompilerEnvWrapper(CompilerEnv, Wrapper):
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

    def close(self):
        self.env.close()

    def reset(self, *args, **kwargs) -> Optional[ObservationType]:
        return self.env.reset(*args, **kwargs)

    def fork(self) -> CompilerEnv:
        return type(self)(env=self.env.fork())

    def step(  # pylint: disable=arguments-differ
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
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
        return self.multistep(
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
        timeout: Optional[float] = 300,
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
        return self.env.multistep(
            actions=actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
        )

    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        return self.env.render(mode)

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.env.reward_range

    @reward_range.setter
    def reward_range(self, value: Tuple[float, float]):
        self.env.reward_range = value

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        self.env.observation_space = observation_space

    @property
    def observation(self) -> ObservationView:
        return self.env.observation

    @observation.setter
    def observation(self, observation: ObservationView) -> None:
        self.env.observation = observation

    @property
    def observation_space_spec(self):
        return self.env.observation_space_spec

    @observation_space_spec.setter
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ) -> None:
        self.env.observation_space_spec = observation_space_spec

    @property
    def reward_space_spec(self) -> Optional[Reward]:
        return self.env.reward_space_spec

    @reward_space_spec.setter
    def reward_space_spec(self, val: Optional[Reward]):
        self.env.reward_space_spec = val

    @property
    def reward_space(self) -> Optional[Reward]:
        return self.env.reward_space

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        self.env.reward_space = reward_space

    @property
    def reward(self) -> RewardView:
        return self.env.reward

    @reward.setter
    def reward(self, reward: RewardView) -> None:
        self.env.reward = reward

    @property
    def action_space(self) -> Space:
        return self.env.action_space

    @action_space.setter
    def action_space(self, action_space: Optional[str]):
        self.env.action_space = action_space

    @property
    def action_spaces(self) -> List[str]:
        return self.env.action_spaces

    @action_spaces.setter
    def action_spaces(self, action_spaces: List[str]):
        self.env.action_spaces = action_spaces

    @property
    def spec(self) -> Any:
        return self.env.spec

    @property
    def benchmark(self) -> Benchmark:
        return self.env.benchmark

    @benchmark.setter
    def benchmark(self, benchmark: Optional[Union[str, Benchmark, BenchmarkUri]]):
        self.env.benchmark = benchmark

    @property
    def datasets(self) -> Iterable[Dataset]:
        return self.env.datasets

    @datasets.setter
    def datasets(self, datasets: Iterable[Dataset]):
        self.env.datasets = datasets

    @property
    def episode_walltime(self) -> float:
        return self.env.episode_walltime

    @property
    def in_episode(self) -> bool:
        return self.env.in_episode

    @property
    def episode_reward(self) -> Optional[float]:
        return self.env.episode_reward

    @episode_reward.setter
    def episode_reward(self, episode_reward: Optional[float]):
        self.env.episode_reward = episode_reward

    @property
    def actions(self) -> List[ActionType]:
        return self.env.actions

    @property
    def version(self) -> str:
        return self.env.version

    @property
    def compiler_version(self) -> str:
        return self.env.compiler_version

    @property
    def state(self) -> CompilerEnvState:
        return self.env.state

    def commandline(self) -> str:
        return self.env.commandline()

    def commandline_to_actions(self, commandline: str) -> List[ActionType]:
        return self.env.commandline_to_actions(commandline)

    def apply(self, state: CompilerEnvState) -> None:  # noqa
        self.env.apply(state)

    def validate(self, state: Optional[CompilerEnvState] = None) -> ValidationResult:
        return self.env.validate(state)


class ActionWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an action space transformation.
    """

    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[ObservationSpaceSpec]] = None,
        reward_spaces: Optional[Iterable[Reward]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
    ):
        return self.env.multistep(
            [self.action(a) for a in actions],
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )

    def action(self, action: ActionType) -> ActionType:
        """Translate the action to the new space."""
        raise NotImplementedError

    def reverse_action(self, action: ActionType) -> ActionType:
        """Translate an action from the new space to the wrapped space."""
        raise NotImplementedError


class ObservationWrapper(CompilerEnvWrapper, ABC):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an observation space transformation.
    """

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        return self.convert_observation(observation)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )

        return self.convert_observation(observation), reward, done, info

    @abstractmethod
    def convert_observation(self, observation: ObservationType) -> ObservationType:
        """Translate an observation to the new space."""
        raise NotImplementedError


class RewardWrapper(CompilerEnvWrapper, ABC):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an reward space transformation.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )

        # Undo the episode_reward update and reapply it once we have transformed
        # the reward.
        #
        # TODO(cummins): Refactor step() so that we don't have to do this
        # recalculation of episode_reward, as this is prone to errors if, say,
        # the base reward returns NaN or an invalid type.
        if reward is not None and self.episode_reward is not None:
            self.unwrapped.episode_reward -= reward
            reward = self.convert_reward(reward)
            self.unwrapped.episode_reward += reward
        return observation, reward, done, info

    @abstractmethod
    def convert_reward(self, reward):
        """Translate a reward to the new space."""
        raise NotImplementedError
