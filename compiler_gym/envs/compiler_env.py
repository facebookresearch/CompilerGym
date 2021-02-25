# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
import logging
import os
import sys
import warnings
from copy import deepcopy
from math import isclose
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fasteners
import gym
import numpy as np
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets.dataset import Dataset, require
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
    ServiceOSError,
    ServiceTransportError,
    observation_t,
)
from compiler_gym.service.proto import (
    AddBenchmarkRequest,
    Benchmark,
    EndSessionReply,
    EndSessionRequest,
    ForkSessionReply,
    ForkSessionRequest,
    GetBenchmarksRequest,
    GetVersionReply,
    GetVersionRequest,
    StartSessionRequest,
    StepRequest,
)
from compiler_gym.spaces import NamedDiscrete, Reward
from compiler_gym.util.debug_util import get_logging_level
from compiler_gym.util.timer import Timer
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView

# Type hints.
info_t = Dict[str, Any]
step_t = Tuple[Optional[observation_t], Optional[float], bool, info_t]


class CompilerEnv(gym.Env):
    """An OpenAI gym environment for compiler optimizations.

    The easiest way to create a CompilerGym environment is to call
    :code:`gym.make()` on one of the registered environments:

    >>> env = gym.make("llvm-v0")

    Alternatively, an environment can be constructed directly, such as by
    connecting to a running compiler service at :code:`localhost:8080` (see
    :doc:`/compiler_gym/service` for more details on connecting to services):

    >>> env = CompilerEnv(
        service="localhost:8080",
        observation_space="features",
        reward_space="runtime",
        rewards=[env_reward_spaces],
    )

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

    :ivar service: A connection to the underlying compiler service.
    :vartype service: compiler_gym.service.CompilerGymServiceConnection

    :ivar logger: A Logger instance used by the environment for communicating
        info and warnings.
    :vartype logger: logging.Logger

    :ivar action_spaces: A list of supported action space names.
    :vartype action_spaces: List[str]

    :ivar reward_range: A tuple indicating the range of reward values.
        Default range is (-inf, +inf).
    :vartype reward_range: Tuple[float, float]

    :ivar datasets_site_path: The filesystem path used by the service
        to store benchmarks.
    :vartype datasets_site_path: Optional[Path]

    :ivar available_datasets: A mapping from dataset name to :class:`Dataset`
        objects that are available to download.
    :vartype available_datasets: Dict[str, Dataset]

    :ivar observation: A view of the available observation spaces that permits
        on-demand computation of observations.
    :vartype observation: compiler_gym.views.ObservationView

    :ivar reward: A view of the available reward spaces that permits on-demand
        computation of rewards.
    :vartype reward: compiler_gym.views.RewardView

    :ivar episode_reward: If
        :func:`CompilerEnv.reward_space <compiler_gym.envs.CompilerGym.reward_space>`
        is set, this value is the sum of all rewards for the current episode.
    :vartype episode_reward: float
    """

    def __init__(
        self,
        service: Union[str, Path],
        rewards: Optional[List[Reward]] = None,
        benchmark: Optional[Union[str, Benchmark]] = None,
        observation_space: Optional[Union[str, ObservationSpaceSpec]] = None,
        reward_space: Optional[Union[str, Reward]] = None,
        action_space: Optional[str] = None,
        connection_settings: Optional[ConnectionOpts] = None,
        service_connection: Optional[CompilerGymServiceConnection] = None,
        logging_level: Optional[int] = None,
    ):
        """Construct and initialize a CompilerGym service environment.

        :param service: The hostname and port of a service that implements the
            CompilerGym service interface, or the path of a binary file
            which provides the CompilerGym service interface when executed.
            See :doc:`/compiler_gym/service` for details.
        :param rewards: The reward spaces that this environment supports.
            Rewards are typically calculated based on observations generated
            by the service. See :class:`Reward <compiler_gym.spaces.Reward>` for
            details.
        :param benchmark: The name of the benchmark to use for this environment.
            The choice of benchmark can be deferred by not providing this
            argument and instead passing by choosing from the
            :code:`CompilerEnv.benchmarks` attribute and passing it to
            :func:`reset()` when called.
        :param observation_space: Compute and return observations at each
            :func:`step()` from this space. Accepts a string name or an
            :class:`ObservationSpaceSpec <compiler_gym.views.ObservationSpaceSpec>`.
            If not provided, :func:`step()` returns :code:`None` for the
            observation value. Can be set later using
            :meth:`env.observation_space <compiler_gym.envs.CompilerEnv.observation_space>`.
            For available spaces, see
            :class:`env.observation.spaces <compiler_gym.views.ObservationView>`.
        :param reward_space: Compute and return reward at each :func:`step()`
            from this space. Accepts a string name or a
            :class:`Reward <compiler_gym.spaces.Reward>`. If
            not provided, :func:`step()` returns :code:`None` for the reward
            value. Can be set later using
            :meth:`env.reward_space <compiler_gym.envs.CompilerEnv.reward_space>`.
            For available spaces, see
            :class:`env.reward.spaces <compiler_gym.views.RewardView>`.
        :param action_space: The name of the action space to use. If not
            specified, the default action space for this compiler is used.
        :param connection_settings: The settings used to establish a connection
            with the remote service.
        :param service_connection: An existing compiler gym service connection
            to use.
        :param logging_level: The integer logging level to use for logging. By
            default, the value reported by
            :func:`get_logging_level() <compiler_gym.get_logging_level>` is
            used.
        :raises FileNotFoundError: If service is a path to a file that is not
            found.
        :raises TimeoutError: If the compiler service fails to initialize
            within the parameters provided in :code:`connection_settings`.
        """
        rewards = rewards or []

        self.metadata = {"render.modes": ["human", "ansi"]}

        # Set up logging.
        self.logger = logging.getLogger("compiler_gym.envs")
        if logging_level is None:
            logging_level = get_logging_level()
        self.logger.setLevel(logging_level)

        # A compiler service supports multiple simultaneous environments. This
        # session ID is used to identify this environment.
        self._session_id: Optional[int] = None

        self._service_endpoint: Union[str, Path] = service
        self._connection_settings = connection_settings or ConnectionOpts()
        self.datasets_site_path: Optional[Path] = None
        self.available_datasets: Dict[str, Dataset] = {}

        # The benchmark that is currently being used, and the benchmark that
        # the user requested. Those do not always correlate, since the user
        # could request a random benchmark.
        self._benchmark_in_use_uri: Optional[str] = benchmark
        self._user_specified_benchmark_uri: Optional[str] = benchmark
        # A map from benchmark URIs to Benchmark messages. We keep track of any
        # user-provided custom benchmarks so that we can register them with a
        # reset service.
        self._custom_benchmarks: Dict[str, Benchmark] = {}

        self.action_space_name = action_space

        self.service = service_connection or CompilerGymServiceConnection(
            endpoint=self._service_endpoint,
            opts=self._connection_settings,
            logger=self.logger,
        )

        # Process the available action, observation, and reward spaces.
        self.action_spaces = [
            self._make_action_space(space.name, space.action)
            for space in self.service.action_spaces
        ]
        self.observation = self._observation_view_type(
            get_observation=lambda req: self.service(self.service.stub.Step, req),
            spaces=self.service.observation_spaces,
        )
        self.reward = self._reward_view_type(rewards, self.observation)

        # Lazily evaluated version strings.
        self._versions: Optional[GetVersionReply] = None

        # Mutable state initialized in reset().
        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None
        self.reward_range: Tuple[float, float] = (-np.inf, np.inf)
        self.episode_reward: Optional[float] = None
        self.episode_start_time: float = time()
        self.actions: List[int] = []

        # Initialize the default observation/reward spaces.
        self._default_observation_space: Optional[ObservationSpaceSpec] = None
        self._default_reward_space: Optional[Reward] = None
        self.observation_space = observation_space
        self.reward_space = reward_space

    @property
    def versions(self) -> GetVersionReply:
        """Get the version numbers from the compiler service."""
        if self._versions is None:
            self._versions = self.service(
                self.service.stub.GetVersion, GetVersionRequest()
            )
        return self._versions

    @property
    def version(self) -> str:
        """The version string of the compiler service."""
        return self.versions.service_version

    @property
    def compiler_version(self) -> str:
        """The version string of the underlying compiler that this service supports."""
        return self.versions.compiler_version

    def commandline(self) -> str:
        """Interface for :class:`CompilerEnv` subclasses to provide an equivalent
        commandline invocation to the current environment state.

        See also
        :meth:`commandline_to_actions() <compiler_gym.envs.CompilerEnv.commandline_to_actions>`.

        Calling this method on a :class:`CompilerEnv` instance raises
        :code:`NotImplementedError`.

        :return: A string commandline invocation.
        """
        raise NotImplementedError("abstract method")

    def commandline_to_actions(self, commandline: str) -> List[int]:
        """Interface for :class:`CompilerEnv` subclasses to convert from a
        commandline invocation to a sequence of actions.

        See also
        :meth:`commandline() <compiler_gym.envs.CompilerEnv.commandline>`.

        Calling this method on a :class:`CompilerEnv` instance raises
        :code:`NotImplementedError`.

        :return: A list of actions.
        """
        raise NotImplementedError("abstract method")

    @property
    def episode_walltime(self) -> float:
        """Return the amount of time in seconds since the last call to
        :meth:`reset() <compiler_env.envs.CompilerEnv.reset>`.
        """
        return time() - self.episode_start_time

    @property
    def state(self) -> CompilerEnvState:
        """The tuple representation of the current environment state."""
        return CompilerEnvState(
            benchmark=self.benchmark,
            reward=self.episode_reward,
            walltime=self.episode_walltime,
            commandline=self.commandline(),
        )

    @property
    def inactive_datasets_site_path(self) -> Optional[Path]:
        """The filesystem path used to store inactive benchmarks."""
        if self.datasets_site_path:
            return (
                self.datasets_site_path.parent
                / f"{self.datasets_site_path.name}.inactive"
            )
        else:
            return None

    @property
    def action_space(self) -> NamedDiscrete:
        """The current action space.

        :getter: Get the current action space.
        :setter: Set the action space to use. Must be an entry in
            :code:`action_spaces`. If :code:`None`, the default action space is
            selected.
        """
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
    def benchmark(self) -> Optional[str]:
        """Get or set the name of the benchmark to use.

        :getter: Get the name of the current benchmark. Returns :code:`None` if
            :func:`__init__` was not provided a benchmark and :func:`reset` has
            not yet been called.
        :setter: Set the benchmark to use. If :code:`None`, a random benchmark
            is selected by the service on each call to :func:`reset`. Else,
            the same benchmark is used for every episode.

        By default, a benchmark will be selected randomly by the service
        from the available :func:`benchmarks` on a call to :func:`reset`. To
        force a specific benchmark to be chosen, set this property (or pass
        the benchmark as an argument to :func:`reset`):

        >>> env.benchmark = "benchmark://foo"
        >>> env.reset()
        >>> env.benchmark
        "benchmark://foo"

        Once set, all subsequent calls to :func:`reset` will select the same
        benchmark.

        >>> env.benchmark = None
        >>> env.reset()  # random benchmark is chosen

        .. note::
            Setting a new benchmark has no effect until :func:`~reset()` is
            called.

        To return to random benchmark selection, set this property to
        :code:`None`:
        """
        return self._benchmark_in_use_uri

    @benchmark.setter
    def benchmark(self, benchmark: Optional[Union[str, Benchmark]]):
        if self.in_episode:
            warnings.warn(
                "Changing the benchmark has no effect until reset() is called."
            )
        if isinstance(benchmark, str) or benchmark is None:
            self._user_specified_benchmark_uri = benchmark
        elif isinstance(benchmark, Benchmark):
            self._user_specified_benchmark_uri = benchmark.uri
            self._add_custom_benchmarks([benchmark])
        else:
            raise TypeError(f"Unsupported benchmark type: {type(benchmark).__name__}")

    @property
    def reward_space(self) -> Optional[Reward]:
        """The default reward space that is used to return a reward value from
        :func:`~step()`.

        :getter: Returns a :class:`Reward <compiler_gym.spaces.Reward>`,
            or :code:`None` if not set.
        :setter: Set the default reward space.
        """
        return self._default_reward_space

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        if isinstance(reward_space, str) and reward_space not in self.reward.spaces:
            raise LookupError(f"Reward space not found: {reward_space}")

        reward_space_name = (
            reward_space.id if isinstance(reward_space, Reward) else reward_space
        )

        self._default_reward: bool = reward_space is not None
        self._default_reward_space: Optional[Reward] = None
        if self._default_reward:
            self._default_reward_space = self.reward.spaces[reward_space_name]
            self.reward_range = (
                self._default_reward_space.min,
                self._default_reward_space.max,
            )
        else:
            self.reward_range = (-np.inf, np.inf)

    @property
    def in_episode(self) -> bool:
        """Whether the service is ready for :func:`step` to be called,
        i.e. :func:`reset` has been called and :func:`close` has not.

        :return: :code:`True` if in an episode, else :code:`False`.
        """
        return self._session_id is not None

    @property
    def observation_space(self) -> Optional[ObservationSpaceSpec]:
        """The observation space that is used to return an observation value in
        :func:`~step()`.

        :getter: Returns the specification of the default observation space, or
            :code:`None` if not set.
        :setter: Set the default observation space.
        """
        return self._default_observation_space

    @observation_space.setter
    def observation_space(
        self, observation_space: Optional[Union[str, ObservationSpaceSpec]]
    ) -> None:
        if (
            isinstance(observation_space, str)
            and observation_space not in self.observation.spaces
        ):
            raise LookupError(f"Observation space not found: {observation_space}")

        observation_space_name = (
            observation_space.id
            if isinstance(observation_space, ObservationSpaceSpec)
            else observation_space
        )

        self._default_observation = observation_space is not None
        self._default_observation_space: Optional[ObservationSpaceSpec] = None
        if self._default_observation:
            self._default_observation_space = self.observation.spaces[
                observation_space_name
            ]

    def fork(self) -> "CompilerEnv":
        """Fork a new environment with exactly the same state.

        This creates a duplicate environment instance with the current state.
        The new environment is entirely independently of the source environment.
        The user must call :meth:`close() <compiler_gym.envs.CompilerEnv.close>`
        on the original and new environments.

        :meth:`reset() <compiler_gym.envs.CompilerEnv.reset>` must be called
        before :code:`fork()`.

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
        if not self.in_episode:
            if self.actions:
                state_to_replay = self.state
                self.logger.warning(
                    "Parent service of fork() has died, replaying state"
                )
            else:
                state_to_replay = None
            self.reset()
            if state_to_replay:
                self.apply(state_to_replay)

        request = ForkSessionRequest(session_id=self._session_id)
        reply: ForkSessionReply = self.service(self.service.stub.ForkSession, request)

        # Create a new environment that shares the connection.
        new_env = type(self)(
            service=self._service_endpoint,
            action_space=self.action_space,
            connection_settings=self._connection_settings,
            service_connection=self.service,
        )

        # Set the session ID.
        new_env._session_id = reply.session_id  # pylint: disable=protected-access
        new_env.observation.session_id = reply.session_id

        # Re-register any custom benchmarks with the new environment.
        if self._custom_benchmarks:
            new_env._add_custom_benchmarks(  # pylint: disable=protected-access
                list(self._custom_benchmarks.values()).copy()
            )

        # Now that we have initialized the environment with the current state,
        # set the benchmark so that calls to new_env.reset() will correctly
        # revert the environment to the initial benchmark state.
        new_env._user_specified_benchmark_uri = (  # pylint: disable=protected-access
            self.benchmark
        )
        # Set the "visible" name of the current benchmark to hide the fact that
        # we loaded from a custom bitcode file.
        new_env._benchmark_in_use_uri = (  # pylint: disable=protected-access
            self.benchmark
        )

        # Create copies of the mutable reward and observation spaces. This
        # is required to correctly calculate incremental updates.
        new_env.reward.spaces = deepcopy(self.reward.spaces)
        new_env.observation.spaces = deepcopy(self.observation.spaces)

        # Set the default observation and reward types. Note the use of IDs here
        # to prevent passing the spaces by reference.
        if self.observation_space:
            new_env.observation_space = self.observation_space.id
        if self.reward_space:
            new_env.reward_space = self.reward_space.id

        # Copy over the mutable episode state.
        new_env.episode_reward = self.episode_reward
        new_env.episode_start_time = self.episode_start_time
        new_env.actions = self.actions.copy()

        return new_env

    def close(self):
        """Close the environment.

        Once closed, :func:`reset` must be called before the environment is used
        again."""
        # Try and close out the episode, but errors are okay.
        close_service = True
        if self.in_episode:
            try:
                reply: EndSessionReply = self.service(
                    self.service.stub.EndSession,
                    EndSessionRequest(session_id=self._session_id),
                )
                # The service still has other sessions attached so we should
                # not kill it.
                if reply.remaining_sessions:
                    close_service = False
            except:  # noqa pylint: disable=bare-except
                pass  # Don't feel bad, computer, you tried ;-)
            self._session_id = None

        if self.service and close_service:
            self.service.close()

        self.service = None

    def __del__(self):
        # Don't let the service be orphaned if user forgot to close(), or
        # if an exception was thrown. The conditional guard is because this
        # may be called in case of early error.
        if hasattr(self, "service") and getattr(self, "service"):
            self.close()

    def reset(  # pylint: disable=arguments-differ
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        retry_count: int = 0,
    ) -> Optional[observation_t]:
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
            subsequent calls to :code:`reset()` will use this action space.
            If no aciton space is provided, the default action space is used.
        :return: The initial observation.
        """
        if retry_count > self._connection_settings.init_max_attempts:
            raise OSError(f"Failed to reset environment after {retry_count} attempts")

        # Start a new service if required.
        if self.service is None:
            self.service = CompilerGymServiceConnection(
                self._service_endpoint, self._connection_settings
            )
            # Re-register the custom benchmarks with the new service.
            self._add_custom_benchmarks(self._custom_benchmarks.values())

        self.action_space_name = action_space or self.action_space_name

        # Stop an existing episode.
        if self.in_episode:
            self.service(
                self.service.stub.EndSession,
                EndSessionRequest(session_id=self._session_id),
            )
            self._session_id = None

        # Update the user requested benchmark, if provided. NOTE: This means
        # that env.reset(benchmark=None) does NOT unset a forced benchmark.
        if benchmark:
            self.benchmark = benchmark

        try:
            reply = self.service(
                self.service.stub.StartSession,
                StartSessionRequest(
                    benchmark=self._user_specified_benchmark_uri,
                    action_space=(
                        [a.name for a in self.action_spaces].index(
                            self.action_space_name
                        )
                        if self.action_space_name
                        else 0
                    ),
                ),
            )
        except (ServiceError, ServiceTransportError):
            # Abort and retry on error.
            self.service.close()
            self.service = None
            return self.reset(
                benchmark=benchmark,
                action_space=action_space,
                retry_count=retry_count + 1,
            )

        self._benchmark_in_use_uri = reply.benchmark
        self._session_id = reply.session_id
        self.observation.session_id = reply.session_id
        self.reward.get_cost = self.observation.__getitem__
        self.episode_start_time = time()
        self.actions = []

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = self._make_action_space(
                self.action_space.name, reply.new_action_space.action
            )

        self.reward.reset(benchmark=self.benchmark)
        if self.reward_space:
            self.episode_reward = 0

        if self.observation_space:
            return self.observation[self.observation_space.id]

    def step(self, action: int) -> step_t:
        """Take a step.

        :param action: Value from the action_space.
        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set. If done
            is True, observation and reward may also be None (e.g. because the
            service failed).
        """
        assert self.in_episode, "Must call reset() before step()"
        observation, reward = None, None

        # Build the list of observations that must be computed by the backend
        # service to generate the user-requested observation and reward.
        # TODO(cummins): We could de-duplicate this list to improve effiency
        # when multiple redundant copies of the same observation space are
        # requested.
        observation_indices, observation_spaces = [], []
        if self.observation_space:
            observation_indices.append(self.observation_space.index)
            observation_spaces.append(self.observation_space.id)
        if self.reward_space:
            observation_indices += [
                self.observation.spaces[obs].index
                for obs in self.reward_space.observation_spaces
            ]
            observation_spaces += self.reward_space.observation_spaces

        # Record the action.
        self.actions.append(action)

        # Send the request to the backend service.
        request = StepRequest(
            session_id=self._session_id,
            action=[action],
            observation_space=observation_indices,
        )
        try:
            reply = self.service(self.service.stub.Step, request)
        except (ServiceError, ServiceTransportError, ServiceOSError, TimeoutError) as e:
            self.close()
            info = {"error_details": str(e)}
            if self.reward_space:
                reward = self.reward_space.reward_on_error(self.episode_reward)
            if self.observation_space:
                observation = self.observation_space.default_value
            return observation, reward, True, info

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = self._make_action_space(
                self.action_space.name, reply.action_space.action
            )

        # Translate observations to python representations.
        if len(reply.observation) != len(observation_indices):
            raise ServiceError(
                f"Requested {observation_indices} observations "
                f"but received {len(reply.observation)}"
            )
        observations = [
            self.observation.spaces[obs].translate(val)
            for obs, val in zip(observation_spaces, reply.observation)
        ]

        # Pop the requested observation.
        if self.observation_space:
            observation, observations = observations[0], observations[1:]

        # Compute reward.
        self.reward.previous_action = action
        if self.reward_space:
            reward = self.reward_space.update(action, observations, self.observation)
            self.episode_reward += reward

        info = {
            "action_had_no_effect": reply.action_had_no_effect,
            "new_action_space": reply.HasField("new_action_space"),
        }

        return observation, reward, reply.end_of_session, info

    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        """Render the environment.

        CompilerEnv instances support two render modes: "human", which prints
        the current environment state to the terminal and return nothing; and
        "ansi", which returns a string representation of the current environment
        state.

        :param mode: The render mode to use.
        :raises TypeError: If a default observation space is not set, or if the
            requested render mode does not exist.
        """
        if not self.observation_space:
            raise ValueError("Cannot call render() when no observation space is used")
        observation = self.observation[self.observation_space.id]
        if mode == "human":
            print(observation)
        elif mode == "ansi":
            return str(observation)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    def benchmarks(self) -> List[str]:
        """Enumerate the list of available benchmarks."""
        reply = self.service(self.service.stub.GetBenchmarks, GetBenchmarksRequest())
        return list(reply.benchmark)

    def _make_action_space(self, name: str, entries: List[str]) -> Space:
        """Create an action space from the given values.

        Subclasses may override this method to produce specialized action
        spaces.

        :param name: The name of the action space.
        :param entries: The entries in the action space.
        :return: A :code:`gym.Space` instance.
        """
        return NamedDiscrete(entries, name)

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

    def require_datasets(self, datasets: List[Union[str, Dataset]]) -> None:
        """Require that the given datasets are available to the environment.

        Example usage:

            >>> env = gym.make("llvm-v0")
            >>> env.require_dataset(["npb-v0"])
            >>> env.benchmarks
            ["npb-v0/1", "npb-v0/2", ...]

        This is equivalent to calling
        :meth:`require(self, dataset) <compiler_gym.datasets.require>` on
        the list of datasets.

        :param datasets: A list of datasets to require. Each dataset is the name
            of an available dataset, the URL of a dataset to download, or a
            :class:`Dataset` instance.
        """
        dataset_installed = False
        for dataset in datasets:
            dataset_installed |= require(self, dataset)
        if dataset_installed:
            # Signal to the compiler service that the contents of the site data
            # directory has changed.
            self.service(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(
                    benchmark=[Benchmark(uri="service://scan-site-data")]
                ),
            )
            self.make_manifest_file()

    def require_dataset(self, dataset: Union[str, Dataset]) -> None:
        """Require that the given dataset is available to the environment.

        Alias for
        :meth:`env.require_datasets([dataset]) <compiler_gym.envs.CompilerEnv.require_datasets>`.

        :param dataset: The name of the dataset to download, the URL of the dataset, or a
            :class:`Dataset` instance.
        """
        return self.require_datasets([dataset])

    def make_manifest_file(self) -> Path:
        """Create the MANIFEST file.

        :return: The path of the manifest file.
        """
        with fasteners.InterProcessLock(self.datasets_site_path / "LOCK"):
            manifest_path = (
                self.datasets_site_path.parent
                / f"{self.datasets_site_path.name}.MANIFEST"
            )
            with open(str(manifest_path), "w") as f:
                for root, _, files in os.walk(self.datasets_site_path):
                    print(
                        "\n".join(
                            [
                                f"{root[len(str(self.datasets_site_path)) + 1:]}/{f}"
                                for f in files
                                if not f.endswith(".json") and f != "LOCK"
                            ]
                        ),
                        file=f,
                    )
        return manifest_path

    def register_dataset(self, dataset: Dataset) -> bool:
        """Register a new dataset.

        After registering, the dataset name may be used by
        :meth:`require_dataset() <compiler_gym.envs.CompilerEnv.require_dataset>`
        to install and activate it.

        Example usage:

            >>> my_dataset = Dataset(name="my-dataset-v0", ...)
            >>> env = gym.make("llvm-v0")
            >>> env.register_dataset(my_dataset)
            >>> env.require_dataset("my-dataset-v0")
            >>> env.benchmark = "my-dataset-v0/1"

        :param dataset: A :class:`Dataset` instance describing the new dataset.
        :return: :code:`True` if the dataset was added, else :code:`False`.
        :raises ValueError: If a dataset with this name is already registered.
        """
        platform = {"darwin": "macos"}.get(sys.platform, sys.platform)
        if platform not in dataset.platforms:
            return False
        if dataset.name in self.available_datasets:
            raise ValueError(f"Dataset already registered with name: {dataset.name}")
        self.available_datasets[dataset.name] = dataset
        return True

    def _add_custom_benchmarks(self, benchmarks: List[Benchmark]) -> None:
        """Register custom benchmarks with the compiler service.

        Benchmark registration occurs automatically using the
        :meth:`env.benchmark <compiler_gym.envs.CompilerEnv.benchmark>`
        property, there is usually no need to call this method yourself.

        :param benchmarks: The benchmarks to register.
        """
        if not benchmarks:
            return

        for benchmark in benchmarks:
            self._custom_benchmarks[benchmark.uri] = benchmark

        self.service(
            self.service.stub.AddBenchmark,
            AddBenchmarkRequest(benchmark=benchmarks),
        )

    def apply(self, state: CompilerEnvState) -> None:  # noqa
        """Replay this state on the given an environment.

        :param env: A :class:`CompilerEnv` instance.
        :raises ValueError: If this state cannot be applied.
        """
        if not self.in_episode:
            self.reset(benchmark=state.benchmark)

        if self.benchmark != state.benchmark:
            warnings.warn(
                f"Applying state from environment for benchmark '{state.benchmark}' "
                f"to environment for benchmark '{self.benchmark}'"
            )

        actions = self.commandline_to_actions(state.commandline)
        for action in actions:
            _, _, done, info = self.step(action)
            if done:
                raise ValueError(
                    f"Environment terminated with error: `{info.get('error_details')}`"
                )

    def validate(self, state: Optional[CompilerEnvState] = None) -> ValidationResult:
        in_place = state is not None
        state = state or self.state

        error_messages = []
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
                        error_messages.append(str(e))
                        break

                    if self.reward_space and self.reward_space.deterministic:
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
                            error_messages.append(
                                f"Expected reward {state.reward:.4f} but "
                                f"received reward {replay_target.episode_reward:.4f}"
                            )

                    # TODO(https://github.com/facebookresearch/CompilerGym/issues/45):
                    # Call the new self.benchmark.validation_callback() method
                    # once implemented.
                    validate_semantics = self.get_benchmark_validation_callback()
                    if validate_semantics:
                        validation["benchmark_semantics_validated"] = True
                        semantics_error = validate_semantics(self)
                        if semantics_error:
                            validation["benchmark_semantics_validation_failed"] = True
                            error_messages.append(semantics_error)

                    # Finished all checks, break the loop.
                    break
        finally:
            fkd.close()

        return ValidationResult(
            walltime=walltime.time,
            error_details="\n".join(error_messages),
            **validation,
        )

    def get_benchmark_validation_callback(
        self,
    ) -> Optional[Callable[["CompilerEnv"], Optional[str]]]:
        """Return a callback that validates benchmark semantics, if available.

        TODO(https://github.com/facebookresearch/CompilerGym/issues/45): This is
        a temporary placeholder for what will eventually become a method on a
        new Benchmark class.
        """
        return None
