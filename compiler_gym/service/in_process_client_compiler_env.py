# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Contains an implementation of the :class:`CompilerEnv<compiler_gym.envs.CompilerEnv>`
interface as a gRPC client service."""
import logging
import numbers
import random
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
from math import isclose
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark, Dataset, Datasets
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.errors import ValidationError
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import ActionSpace as ActionSpaceProto
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import Event, GetVersionReply
from compiler_gym.service.proto import NamedDiscreteSpace as NamedDiscreteSpaceProto
from compiler_gym.service.proto import ObservationSpace as ObservationSpaceProto
from compiler_gym.service.proto import Space as SpaceProto
from compiler_gym.service.proto import py_converters
from compiler_gym.spaces import (
    ActionSpace,
    DefaultRewardFromObservation,
    NamedDiscrete,
    Reward,
)
from compiler_gym.util.gym_type_hints import (
    ActionType,
    ObservationType,
    OptionalArgumentValue,
    RewardType,
    StepType,
)
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.timer import Timer
from compiler_gym.util.version import __version__
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView

logger = logging.getLogger(__name__)


class ServiceMessageConverters:
    """Allows for customization of conversion to/from gRPC messages for the
    :class:`InProcessClientCompilerEnv
    <compiler_gym.service.client_service_compiler_env.InProcessClientCompilerEnv>`.

    Supports conversion customizations:

        - :code:`compiler_gym.service.proto.ActionSpace` ->
          :code:`gym.spaces.Space`.
        - :code:`compiler_gym.util.gym_type_hints.ActionType` ->
          :code:`compiler_gym.service.proto.Event`.
    """

    action_space_converter: Callable[[ActionSpaceProto], ActionSpace]
    action_converter: Callable[[ActionType], Event]

    def __init__(
        self,
        action_space_converter: Optional[
            Callable[[ActionSpaceProto], ActionSpace]
        ] = None,
        action_converter: Optional[Callable[[Any], Event]] = None,
    ):
        """Constructor."""
        self.action_space_converter = (
            py_converters.make_action_space_wrapper(
                py_converters.make_message_default_converter()
            )
            if action_space_converter is None
            else action_space_converter
        )
        self.action_converter = (
            py_converters.to_event_message_default_converter()
            if action_converter is None
            else action_converter
        )


def make_working_directory(session_type: Type[CompilationSession]) -> Path:
    random_hash = random.getrandbits(16)
    timestamp = datetime.now().strftime(f"s/%m%dT%H%M%S-%f-{random_hash:04x}")
    working_directory = transient_cache_path(f"s/{session_type.__name__}-{timestamp}")
    logger.debug(
        "Created working directory for compilation session: %s", working_directory
    )
    return working_directory


def action_space_to_proto(action_space: ActionSpace) -> ActionSpaceProto:
    # TODO(cummins): This needs to be a true reverse mapping from python to
    # proto. Currently it's hardcoded to work only for named discrete spaces.
    return ActionSpaceProto(
        name=action_space.name,
        space=SpaceProto(
            named_discrete=NamedDiscreteSpaceProto(name=action_space.names)
        ),
    )


class InProcessClientCompilerEnv(CompilerEnv):
    """Implementation of :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
    for Python services that run in the same process.

    This uses the same protocol buffer interface as
    :class:`InProcessClientCompilerEnv
    <compiler_gym.service.InProcessClientCompilerEnv>`, but without the overhead
    of running a gRPC service. The tradeoff is reduced robustness in the face of
    compiler errors, and the inability to run the service on a different
    machine.
    """

    def __init__(
        self,
        session_type: Type[CompilationSession],
        session: Optional[CompilationSession] = None,
        rewards: Optional[List[Reward]] = None,
        datasets: Optional[Iterable[Dataset]] = None,
        benchmark: Optional[Union[str, Benchmark]] = None,
        observation_space: Optional[Union[str, ObservationSpaceSpec]] = None,
        reward_space: Optional[Union[str, Reward]] = None,
        action_space: Optional[str] = None,
        derived_observation_spaces: Optional[List[Dict[str, Any]]] = None,
        service_message_converters: ServiceMessageConverters = None,
    ):
        """Construct and initialize a CompilerGym environment.

        In normal use you should use :code:`gym.make(...)` rather than calling
        the constructor directly.

        :param rewards: The reward spaces that this environment supports.
            Rewards are typically calculated based on observations generated by
            the service. See :class:`Reward <compiler_gym.spaces.Reward>` for
            details.

        :param benchmark: The benchmark to use for this environment. Either a
            URI string, or a :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instance. If not provided, the
            first benchmark as returned by
            :code:`next(env.datasets.benchmarks())` will be used as the default.

        :param observation_space: Compute and return observations at each
            :func:`step()` from this space. Accepts a string name or an
            :class:`ObservationSpaceSpec
            <compiler_gym.views.ObservationSpaceSpec>`. If not provided,
            :func:`step()` returns :code:`None` for the observation value. Can
            be set later using :meth:`env.observation_space
            <compiler_gym.envs.InProcessClientCompilerEnv.observation_space>`. For available
            spaces, see :class:`env.observation.spaces
            <compiler_gym.views.ObservationView>`.

        :param reward_space: Compute and return reward at each :func:`step()`
            from this space. Accepts a string name or a :class:`Reward
            <compiler_gym.spaces.Reward>`. If not provided, :func:`step()`
            returns :code:`None` for the reward value. Can be set later using
            :meth:`env.reward_space
            <compiler_gym.envs.InProcessClientCompilerEnv.reward_space>`. For available spaces,
            see :class:`env.reward.spaces <compiler_gym.views.RewardView>`.

        :param action_space: The name of the action space to use. If not
            specified, the default action space for this compiler is used.

        :param derived_observation_spaces: An optional list of arguments to be
            passed to :meth:`env.observation.add_derived_space()
            <compiler_gym.views.observation.Observation.add_derived_space>`.

        :param service_message_converters: Custom converters for action spaces and actions.

        :param connection_settings: The settings used to establish a connection
            with the remote service.

        :param service_connection: An existing compiler gym service connection
            to use.

        :raises FileNotFoundError: If service is a path to a file that is not
            found.

        :raises TimeoutError: If the compiler service fails to initialize within
            the parameters provided in :code:`connection_settings`.
        """
        self.session_type = session_type

        self.metadata = {"render.modes": ["human", "ansi"]}

        self._datasets = Datasets(datasets or [])

        self.action_space_name = action_space

        # If no reward space is specified, generate some from numeric observation spaces
        rewards = rewards or [
            DefaultRewardFromObservation(obs.name)
            for obs in self.session_type.observation_spaces
            if obs.default_observation.WhichOneof("value")
            and isinstance(
                getattr(
                    obs.default_observation, obs.default_observation.WhichOneof("value")
                ),
                numbers.Number,
            )
        ]

        # The benchmark that is currently being used, and the benchmark that
        # will be used on the next call to reset(). These are equal except in
        # the gap between the user setting the env.benchmark property while in
        # an episode and the next call to env.reset().
        self._benchmark_in_use: Optional[Benchmark] = None
        self._benchmark_in_use_proto: BenchmarkProto = BenchmarkProto()
        self._next_benchmark: Optional[Benchmark] = None
        # Normally when the benchmark is changed the updated value is not
        # reflected until the next call to reset(). We make an exception for the
        # constructor-time benchmark as otherwise the behavior of the benchmark
        # property is counter-intuitive:
        #
        #     >>> env = gym.make("example-v0", benchmark="foo")
        #     >>> env.benchmark
        #     None
        #     >>> env.reset()
        #     >>> env.benchmark
        #     "foo"
        #
        # By forcing the _benchmark_in_use URI at constructor time, the first
        # env.benchmark above returns the benchmark as expected.
        try:
            self.benchmark = benchmark or next(self.datasets.benchmarks())
            self._benchmark_in_use = self._next_benchmark
        except StopIteration:
            # StopIteration raised on next(self.datasets.benchmarks()) if there
            # are no benchmarks available. This is to allow InProcessClientCompilerEnv to be
            # used without any datasets by setting a benchmark before/during the
            # first reset() call.
            pass

        self.service_message_converters = (
            ServiceMessageConverters()
            if service_message_converters is None
            else service_message_converters
        )

        # Process the available action, observation, and reward spaces.
        self.action_spaces = [
            self.service_message_converters.action_space_converter(space)
            for space in self.session_type.action_spaces
        ]

        self.observation = self._observation_view_type(
            raw_step=self.multistep,
            spaces=self.session_type.observation_spaces,
        )
        self.reward = self._reward_view_type(rewards, self.observation)

        # Register any derived observation spaces now so that the observation
        # space can be set below.
        for derived_observation_space in derived_observation_spaces or []:
            self.observation.add_derived_space_internal(**derived_observation_space)

        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None

        # Mutable state initialized in reset().
        self._reward_range: Tuple[float, float] = (-np.inf, np.inf)
        self.episode_reward = None
        self.episode_start_time: float = time()
        self._actions: List[ActionType] = []

        # Initialize the default observation/reward spaces.
        self.observation_space_spec = None
        self.reward_space_spec = None
        self.observation_space = observation_space
        self.reward_space = reward_space

        self.working_directory: Optional[Path] = None
        self.session: Optional[CompilationSession] = session

    def close(self):
        if self.working_directory:
            shutil.rmtree(self.working_directory, ignore_errors=True)
            self.working_directory = None

    def __del__(self):
        # Don't let the service be orphaned if user forgot to close(), or
        # if an exception was thrown. The conditional guard is because this
        # may be called in case of early error.
        if hasattr(self, "service") and getattr(self, "service"):
            self.close()

    @property
    def observation_space_spec(self) -> ObservationSpaceSpec:
        return self._observation_space_spec

    @observation_space_spec.setter
    def observation_space_spec(
        self, observation_space_spec: Optional[ObservationSpaceSpec]
    ):
        self._observation_space_spec = observation_space_spec

    @property
    def observation(self) -> ObservationView:
        return self._observation

    @observation.setter
    def observation(self, observation: ObservationView) -> None:
        self._observation = observation

    @property
    def reward_space_spec(self) -> Optional[Reward]:
        return self._reward_space_spec

    @reward_space_spec.setter
    def reward_space_spec(self, val: Optional[Reward]):
        self._reward_space_spec = val

    @property
    def datasets(self) -> Iterable[Dataset]:
        return self._datasets

    @datasets.setter
    def datasets(self, datasets: Iterable[Dataset]):
        self._datastes = datasets

    @property
    def episode_reward(self) -> Optional[float]:
        return self._episode_reward

    @episode_reward.setter
    def episode_reward(self, episode_reward: Optional[float]):
        self._episode_reward = episode_reward

    @property
    def actions(self) -> List[ActionType]:
        return self._actions

    @property
    def versions(self) -> GetVersionReply:
        """Get the version numbers from the compiler service."""
        return GetVersionReply(
            service_version=__version__,
            compiler_version=self.session_type.compiler_version,
        )

    @property
    def version(self) -> str:
        """The version string of the compiler service."""
        return self.versions.service_version

    @property
    def compiler_version(self) -> str:
        """The version string of the underlying compiler that this service supports."""
        return self.versions.compiler_version

    @property
    def episode_walltime(self) -> float:
        return time() - self.episode_start_time

    @property
    def state(self) -> CompilerEnvState:
        return CompilerEnvState(
            benchmark=str(self.benchmark) if self.benchmark else None,
            reward=self.episode_reward,
            walltime=self.episode_walltime,
            commandline=self.action_space.to_string(self.actions),
        )

    @property
    def action_space(self) -> Space:
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: Optional[str]):
        self.action_space_name = action_space
        index = (
            [a.name for a in self.action_spaces].index(action_space)
            if self.action_space_name
            else 0
        )
        self._action_space: NamedDiscrete = self.action_spaces[index]

    @property
    def action_spaces(self) -> List[str]:
        return self._action_spaces

    @action_spaces.setter
    def action_spaces(self, action_spaces: List[str]):
        self._action_spaces = action_spaces

    @property
    def benchmark(self) -> Benchmark:
        return self._benchmark_in_use

    @benchmark.setter
    def benchmark(self, benchmark: Union[str, Benchmark, BenchmarkUri]):
        warnings.warn("Changing the benchmark has no effect until reset() is called")
        if isinstance(benchmark, str):
            benchmark_object = self.datasets.benchmark(benchmark)
            logger.debug("Setting benchmark by name: %s", benchmark_object)
            self._next_benchmark = benchmark_object
        elif isinstance(benchmark, Benchmark):
            logger.debug("Setting benchmark: %s", benchmark.uri)
            self._next_benchmark = benchmark
        elif isinstance(benchmark, BenchmarkUri):
            benchmark_object = self.datasets.benchmark_from_parsed_uri(benchmark)
            logger.debug("Setting benchmark by name: %s", benchmark_object)
            self._next_benchmark = benchmark_object
        else:
            raise TypeError(
                f"Expected a Benchmark or str, received: '{type(benchmark).__name__}'"
            )

    @property
    def reward_space(self) -> Optional[Reward]:
        return self.reward_space_spec

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        # Coerce the observation space into a string.
        reward_space: Optional[str] = (
            reward_space.name if isinstance(reward_space, Reward) else reward_space
        )

        if reward_space:
            if reward_space not in self.reward.spaces:
                raise LookupError(f"Reward space not found: {reward_space}")
            # The reward space remains unchanged, nothing to do.
            if reward_space == self.reward_space:
                return
            self.reward_space_spec = self.reward.spaces[reward_space]
            self._reward_range = (
                self.reward_space_spec.min,
                self.reward_space_spec.max,
            )
            # Reset any cumulative rewards.
            self.episode_reward = 0
        else:
            # If no reward space is being used then set the reward range to
            # unbounded.
            self.reward_space_spec = None
            self._reward_range = (-np.inf, np.inf)

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self._reward_range

    @property
    def reward(self) -> RewardView:
        return self._reward

    @reward.setter
    def reward(self, reward: RewardView) -> None:
        self._reward = reward

    @property
    def observation_space(self) -> Optional[Space]:
        if self.observation_space_spec:
            return self.observation_space_spec.space

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        # Coerce the observation space into a string.
        observation_space: Optional[str] = (
            observation_space.id
            if isinstance(observation_space, ObservationSpaceSpec)
            else observation_space
        )

        if observation_space:
            if observation_space not in self.observation.spaces:
                raise LookupError(f"Observation space not found: {observation_space}")
            self.observation_space_spec = self.observation.spaces[observation_space]
        else:
            self.observation_space_spec = None

    def _init_kwargs(self) -> Dict[str, Any]:
        """Retturn a dictionary of keyword arguments used to initialize the
        environment.
        """
        return {
            "session_type": self.session_type,
            "action_space": self.action_space,
            "benchmark": self.benchmark,
            "connection_settings": self._connection_settings,
        }

    def fork(self) -> "InProcessClientCompilerEnv":
        try:
            new_session: CompilationSession = self.session.fork()

            # Create a new environment that shares the connection.
            new_env = type(self)(**self._init_kwargs(), session=new_session)

            # Now that we have initialized the environment with the current
            # state, set the benchmark so that calls to new_env.reset() will
            # correctly revert the environment to the initial benchmark state.
            #
            # pylint: disable=protected-access
            new_env._next_benchmark = self._benchmark_in_use

            # Set the "visible" name of the current benchmark to hide the fact
            # that we loaded from a custom benchmark file.
            new_env._benchmark_in_use = self._benchmark_in_use
        except NotImplementedError:
            # Fallback implementation. If the compiler service does not support
            # the Fork() operator then we create a new independent environment
            # and apply the sequence of actions in the current environment to
            # replay the state.
            new_env = type(self)(**self._init_kwargs())
            new_env.reset()
            _, _, done, _ = new_env.multistep(self.actions)
            assert not done, "Failed to replay action sequence in forked environment"

        # Create copies of the mutable reward and observation spaces. This
        # is required to correctly calculate incremental updates.
        new_env.reward.spaces = deepcopy(self.reward.spaces)
        new_env.observation.spaces = deepcopy(self.observation.spaces)

        # Set the default observation and reward types. Note the use of IDs here
        # to prevent passing the spaces by reference.
        if self.observation_space:
            new_env.observation_space = self.observation_space_spec.id
        if self.reward_space:
            new_env.reward_space = self.reward_space.name

        # Copy over the mutable episode state.
        new_env.episode_reward = self.episode_reward
        new_env.episode_start_time = self.episode_start_time
        new_env._actions = self.actions.copy()

        return new_env

    def reset(
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        reward_space: Union[
            OptionalArgumentValue, str, Reward
        ] = OptionalArgumentValue.UNCHANGED,
        observation_space: Union[
            OptionalArgumentValue, str, ObservationSpaceSpec
        ] = OptionalArgumentValue.UNCHANGED,
        timeout: Optional[float] = 300,
    ) -> Optional[ObservationType]:
        shutil.rmtree(self.working_directory, ignore_errors=True)
        self.working_directory = make_working_directory(self.session_type)

        if observation_space != OptionalArgumentValue.UNCHANGED:
            self.observation_space = observation_space

        if reward_space != OptionalArgumentValue.UNCHANGED:
            self.reward_space = reward_space

        if not self._next_benchmark:
            raise TypeError(
                "No benchmark set. Set a benchmark using "
                "`env.reset(benchmark=benchmark)`. Use `env.datasets` to "
                "access the available benchmarks."
            )

        self.action_space_name = action_space or self.action_space_name

        # Update the user requested benchmark, if provided.
        if benchmark:
            self.benchmark = benchmark
        self._benchmark_in_use = self._next_benchmark
        self._benchmark_in_use_proto = self._benchmark_in_use.proto

        self.session = self.session_type(
            working_directory=self.working_directory,
            action_space=action_space_to_proto(self.action_space),
            benchmark=self._benchmark_in_use_proto,
        )

        self.reward.get_cost = self.observation.__getitem__
        self.episode_start_time = time()
        self._actions = []

        self.reward.reset(benchmark=self.benchmark, observation_view=self.observation)
        if self.reward_space:
            self.episode_reward = 0.0

        if self.observation_space:
            return self.observation.spaces[self.observation_space_spec.id].translate(
                self.session.get_observation(
                    ObservationSpaceProto(name=self.observation_space_spec.id)
                )
            )

    @property
    def in_episode(self) -> bool:
        return self.session is not None

    def step(
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
    ) -> StepType:
        """:raises SessionNotFound: If :meth:`reset()
        <compiler_gym.envs.InProcessClientCompilerEnv.reset>` has not been called.
        """
        return self.multistep(
            [action], observation_spaces, reward_spaces, timeout=timeout
        )

    def multistep(
        self,
        actions: Iterable[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        timeout: Optional[float] = 300,
    ):
        """:raises SessionNotFound: If :meth:`reset()
        <compiler_gym.envs.InProcessClientCompilerEnv.reset>` has not been called.
        """
        # Coerce observation spaces into a list of ObservationSpaceSpec instances.
        if observation_spaces:
            observation_spaces_to_compute: List[ObservationSpaceSpec] = [
                obs
                if isinstance(obs, ObservationSpaceSpec)
                else self.observation.spaces[obs]
                for obs in observation_spaces
            ]
        elif self.observation_space_spec:
            observation_spaces_to_compute: List[ObservationSpaceSpec] = [
                self.observation_space_spec
            ]
            observation_spaces = [self.observation_space_spec]
        else:
            observation_spaces_to_compute: List[ObservationSpaceSpec] = []
            observation_spaces = []

        # Coerce reward spaces into a list of Reward instances.
        if reward_spaces:
            reward_spaces_to_compute: List[Reward] = [
                rew if isinstance(rew, Reward) else self.reward.spaces[rew]
                for rew in reward_spaces
            ]
        elif self.reward_space:
            reward_spaces_to_compute: List[Reward] = [self.reward_space]
            reward_spaces = [self.reward_space]
        else:
            reward_spaces_to_compute: List[Reward] = []
            reward_spaces = []

        reward_observation_spaces: List[ObservationSpaceSpec] = []
        for reward_space in reward_spaces:
            reward_observation_spaces += [
                self.observation.spaces[obs] for obs in reward_space.observation_spaces
            ]

        observations_to_compute: List[ObservationSpaceSpec] = list(
            set(observation_spaces).union(set(reward_observation_spaces))
        )

        # Record the actions.
        self._actions += actions

        done, new_action_space, action_had_no_effect = False, False, True
        for action in actions:
            (
                done,
                new_new_action_space,
                new_action_had_no_effect,
            ) = self.session.apply_action(
                self.service_message_converters.action_converter(action)
            )
            new_action_space |= new_new_action_space is not None
            action_had_no_effect &= new_action_had_no_effect

            # If the action space has changed, update it.
            if new_new_action_space:
                self._action_space = (
                    self.service_message_converters.action_space_converter(
                        new_new_action_space
                    )
                )

            if done:
                default_observations = [
                    observation_space.default_value
                    for observation_space in observation_spaces
                ]
                default_rewards = [
                    float(reward_space.reward_on_error(self.episode_reward))
                    for reward_space in reward_spaces
                ]
                return (
                    default_observations,
                    default_rewards,
                    True,
                    {
                        "episode_ended_by_environment": True,
                    },
                )

        # Translate observations to python representations.
        computed_observations = {
            observation_space.id: observation_space.translate(
                self.session.get_observation(
                    ObservationSpaceProto(name=observation_space.id)
                )
            )
            for observation_space in observations_to_compute
        }

        # Get the user-requested observation.
        observations: List[ObservationType] = [
            computed_observations[observation_space.id]
            for observation_space in observation_spaces
        ]

        # Update and compute the rewards.
        rewards: List[RewardType] = []
        for reward_space in reward_spaces:
            reward_observations = [
                computed_observations[observation_space]
                for observation_space in reward_space.observation_spaces
            ]
            rewards.append(
                float(
                    reward_space.update(actions, reward_observations, self.observation)
                )
            )

        info = {
            "action_had_no_effect": action_had_no_effect,
            "new_action_space": new_action_space,
        }

        # Translate observations lists back to the appropriate types.
        if observation_spaces is None and self.observation_space_spec:
            observations = observations[0]
        elif not observation_spaces_to_compute:
            observations = None

        # Translate reward lists back to the appropriate types.
        if reward_spaces is None and self.reward_space:
            rewards = rewards[0]
            # Update the cumulative episode reward
            self.episode_reward += rewards
        elif not reward_spaces_to_compute:
            rewards = None

        return observations, rewards, done, info

    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        """Render the environment.

        InProcessClientCompilerEnv instances support two render modes: "human", which prints
        the current environment state to the terminal and return nothing; and
        "ansi", which returns a string representation of the current environment
        state.

        :param mode: The render mode to use.
        :raises TypeError: If a default observation space is not set, or if the
            requested render mode does not exist.
        """
        if not self.observation_space:
            raise ValueError("Cannot call render() when no observation space is used")
        observation = self.observation[self.observation_space_spec.id]
        if mode == "human":
            print(observation)
        elif mode == "ansi":
            return str(observation)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    def _observation_view_type(self):
        """Returns the type for observation views.

        Subclasses may override this to extend the default observation view.
        """
        return ObservationView

    @property
    def _reward_view_type(self):
        """Returns the type for reward views.

        Subclasses may override this to extend the default reward view.
        """
        return RewardView

    def apply(self, state: CompilerEnvState) -> None:  # noqa
        # TODO(cummins): Does this behavior make sense? Take, for example:
        #
        #     >>> env.apply(state)
        #     >>> env.benchmark == state.benchmark
        #     False
        #
        # I think most users would reasonable expect `env.apply(state)` to fully
        # apply the state, not just the sequence of actions. And what about the
        # choice of observation space, reward space, etc?
        if self.benchmark != state.benchmark:
            warnings.warn(
                f"Applying state from environment for benchmark '{state.benchmark}' "
                f"to environment for benchmark '{self.benchmark}'"
            )
            self.reset(benchmark=state.benchmark)

        actions = self.action_space.from_string(state.commandline)
        done = False
        for action in actions:
            _, _, done, info = self.step(action)
        if done:
            raise ValueError(
                f"Environment terminated with error: `{info.get('error_details')}`"
            )

    def validate(self, state: Optional[CompilerEnvState] = None) -> ValidationResult:
        if state:
            self.reset(benchmark=state.benchmark)
            in_place = False
        else:
            state = self.state
            in_place = True

        errors: ValidationError = []
        validation = {
            "state": state,
            "actions_replay_failed": False,
            "reward_validated": False,
            "reward_validation_failed": False,
            "benchmark_semantics_validated": False,
            "benchmark_semantics_validation_failed": False,
        }

        fkd = self.fork()
        try:
            with Timer() as walltime:
                replay_target = self if in_place else fkd
                replay_target.reset(benchmark=state.benchmark)
                # Use a while loop here so that we can `break` early out of the
                # validation process in case a step fails.
                while True:
                    try:
                        replay_target.apply(state)
                    except (ValueError, OSError) as e:
                        validation["actions_replay_failed"] = True
                        errors.append(
                            ValidationError(
                                type="Action replay failed",
                                data={
                                    "exception": str(e),
                                    "exception_type": type(e).__name__,
                                },
                            )
                        )
                        break

                    if state.reward is not None and self.reward_space is None:
                        warnings.warn(
                            "Validating state with reward, but "
                            "environment has no reward space set"
                        )
                    elif (
                        state.reward is not None
                        and self.reward_space
                        and self.reward_space.deterministic
                    ):
                        validation["reward_validated"] = True
                        # If reward deviates from the expected amount record the
                        # error but continue with the remainder of the validation.
                        if not isclose(
                            state.reward,
                            replay_target.episode_reward,
                            rel_tol=1e-5,
                            abs_tol=1e-10,
                        ):
                            validation["reward_validation_failed"] = True
                            errors.append(
                                ValidationError(
                                    type=(
                                        f"Expected reward {state.reward} but "
                                        f"received reward {replay_target.episode_reward}"
                                    ),
                                    data={
                                        "expected_reward": state.reward,
                                        "actual_reward": replay_target.episode_reward,
                                    },
                                )
                            )

                    benchmark = replay_target.benchmark
                    if benchmark.is_validatable():
                        validation["benchmark_semantics_validated"] = True
                        semantics_errors = benchmark.validate(replay_target)
                        if semantics_errors:
                            validation["benchmark_semantics_validation_failed"] = True
                            errors += semantics_errors

                    # Finished all checks, break the loop.
                    break
        finally:
            fkd.close()

        return ValidationResult.construct(
            walltime=walltime.time,
            errors=errors,
            **validation,
        )

    def send_param(self, key: str, value: str) -> str:
        """Send a single <key, value> parameter to the compiler service.

        See :meth:`send_params()
        <compiler_gym.envs.InProcessClientCompilerEnv.send_params>` for more
        information.

        :param key: The parameter key.

        :param value: The parameter value.

        :return: The response from the compiler service.

        :raises SessionNotFound: If called before :meth:`reset()
            <compiler_gym.envs.InProcessClientCompilerEnv.reset>`.
        """
        return self.session.handle_session_parameter(key, value)

    def send_params(self, *params: Iterable[Tuple[str, str]]) -> List[str]:
        """Send a list of <key, value> parameters to the compiler service.

        This provides a mechanism to send messages to the backend compilation
        session in a way that doesn't conform to the normal communication
        pattern. This can be useful for things like configuring runtime
        debugging settings, or applying "meta actions" to the compiler that are
        not exposed in the compiler's action space. Consult the documentation
        for a specific compiler service to see what parameters, if any, are
        supported.

        Must have called :meth:`reset()
        <compiler_gym.envs.InProcessClientCompilerEnv.reset>` first.

        :param params: A list of parameters, where each parameter is a
            :code:`(key, value)` tuple.

        :return: A list of string responses, one per parameter.

        :raises SessionNotFound: If called before :meth:`reset()
            <compiler_gym.envs.InProcessClientCompilerEnv.reset>`.
        """
        return [
            self.session.handle_session_parameter(key, value) for key, value in params
        ]

    def __copy__(self) -> "InProcessClientCompilerEnv":
        raise TypeError(
            "InProcessClientCompilerEnv instances do not support shallow copies. Use deepcopy()"
        )

    def __deepcopy__(self, memo) -> "InProcessClientCompilerEnv":
        del memo  # unused
        return self.fork()
