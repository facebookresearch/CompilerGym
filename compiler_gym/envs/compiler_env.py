# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
import logging
import numbers
import warnings
from collections.abc import Iterable as IterableType
from copy import deepcopy
from math import isclose
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
from deprecated.sphinx import deprecated
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark, Dataset, Datasets
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
    ServiceOSError,
    ServiceTransportError,
    SessionNotFound,
)
from compiler_gym.service.proto import (
    AddBenchmarkRequest,
    EndSessionReply,
    EndSessionRequest,
    ForkSessionReply,
    ForkSessionRequest,
    GetVersionReply,
    GetVersionRequest,
    StartSessionRequest,
    StepReply,
    StepRequest,
)
from compiler_gym.spaces import DefaultRewardFromObservation, NamedDiscrete, Reward
from compiler_gym.util.debug_util import get_logging_level
from compiler_gym.util.gym_type_hints import ObservationType, StepType
from compiler_gym.util.timer import Timer
from compiler_gym.validation_error import ValidationError
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView


def _wrapped_step(
    service: CompilerGymServiceConnection, request: StepRequest
) -> StepReply:
    """Call the Step() RPC endpoint."""
    try:
        return service(service.stub.Step, request)
    except FileNotFoundError as e:
        if str(e).startswith("Session not found"):
            raise SessionNotFound(str(e))
        raise


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
        datasets: Optional[Iterable[Dataset]] = None,
        benchmark: Optional[Union[str, Benchmark]] = None,
        observation_space: Optional[Union[str, ObservationSpaceSpec]] = None,
        reward_space: Optional[Union[str, Reward]] = None,
        action_space: Optional[str] = None,
        connection_settings: Optional[ConnectionOpts] = None,
        service_connection: Optional[CompilerGymServiceConnection] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Construct and initialize a CompilerGym service environment.

        :param service: The hostname and port of a service that implements the
            CompilerGym service interface, or the path of a binary file which
            provides the CompilerGym service interface when executed. See
            :doc:`/compiler_gym/service` for details.

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
            <compiler_gym.envs.CompilerEnv.observation_space>`. For available
            spaces, see :class:`env.observation.spaces
            <compiler_gym.views.ObservationView>`.

        :param reward_space: Compute and return reward at each :func:`step()`
            from this space. Accepts a string name or a :class:`Reward
            <compiler_gym.spaces.Reward>`. If not provided, :func:`step()`
            returns :code:`None` for the reward value. Can be set later using
            :meth:`env.reward_space
            <compiler_gym.envs.CompilerEnv.reward_space>`. For available spaces,
            see :class:`env.reward.spaces <compiler_gym.views.RewardView>`.

        :param action_space: The name of the action space to use. If not
            specified, the default action space for this compiler is used.

        :param connection_settings: The settings used to establish a connection
            with the remote service.

        :param service_connection: An existing compiler gym service connection
            to use.

        :param logger: The logger to use for this environment. If not provided,
            a :code:`compiler_gym.envs` logger is used and assigned the
            verbosity returned by :func:`get_logging_level()
            <compiler_gym.get_logging_level>`.

        :raises FileNotFoundError: If service is a path to a file that is not
            found.

        :raises TimeoutError: If the compiler service fails to initialize within
            the parameters provided in :code:`connection_settings`.
        """
        self.metadata = {"render.modes": ["human", "ansi"]}

        if logger is None:
            logger = logging.getLogger("compiler_gym.envs")
            logger.setLevel(get_logging_level())
        self.logger = logger

        # A compiler service supports multiple simultaneous environments. This
        # session ID is used to identify this environment.
        self._session_id: Optional[int] = None

        self._service_endpoint: Union[str, Path] = service
        self._connection_settings = connection_settings or ConnectionOpts()

        self.action_space_name = action_space

        self.service = service_connection or CompilerGymServiceConnection(
            endpoint=self._service_endpoint,
            opts=self._connection_settings,
            logger=self.logger,
        )
        self.datasets = Datasets(datasets or [])

        # If no reward space is specified, generate some from numeric observation spaces
        rewards = rewards or [
            DefaultRewardFromObservation(obs.name)
            for obs in self.service.observation_spaces
            if obs.default_value.WhichOneof("value")
            and isinstance(
                getattr(obs.default_value, obs.default_value.WhichOneof("value")),
                numbers.Number,
            )
        ]

        # The benchmark that is currently being used, and the benchmark that
        # will be used on the next call to reset(). These are equal except in
        # the gap between the user setting the env.benchmark property while in
        # an episode and the next call to env.reset().
        self._benchmark_in_use: Optional[Benchmark] = None
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
            # are no benchmarks available. This is to allow CompilerEnv to be
            # used without any datasets by setting a benchmark before/during the
            # first reset() call.
            pass

        # Process the available action, observation, and reward spaces.
        self.action_spaces = [
            self._make_action_space(space.name, space.action)
            for space in self.service.action_spaces
        ]
        self.observation = self._observation_view_type(
            get_observation=lambda req: _wrapped_step(self.service, req),
            spaces=self.service.observation_spaces,
        )
        self.reward = self._reward_view_type(rewards, self.observation)

        # Lazily evaluated version strings.
        self._versions: Optional[GetVersionReply] = None

        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None

        # Mutable state initialized in reset().
        self.reward_range: Tuple[float, float] = (-np.inf, np.inf)
        self.episode_reward: Optional[float] = None
        self.episode_start_time: float = time()
        self.actions: List[int] = []

        # Initialize the default observation/reward spaces.
        self.observation_space_spec: Optional[ObservationSpaceSpec] = None
        self.reward_space_spec: Optional[Reward] = None
        self.observation_space = observation_space
        self.reward_space = reward_space

    @property
    @deprecated(
        version="0.1.8",
        reason=(
            "Use :meth:`env.datasets.datasets() <compiler_gym.datasets.Datasets.datasets>` instead. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def available_datasets(self) -> Dict[str, Dataset]:
        """A dictionary of datasets."""
        return {d.name: d for d in self.datasets}

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
        """Interface for :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
        subclasses to provide an equivalent commandline invocation to the
        current environment state.

        See also :meth:`commandline_to_actions()
        <compiler_gym.envs.CompilerEnv.commandline_to_actions>`.

        Calling this method on a :class:`CompilerEnv
        <compiler_gym.envs.CompilerEnv>` instance raises
        :code:`NotImplementedError`.

        :return: A string commandline invocation.
        """
        raise NotImplementedError("abstract method")

    def commandline_to_actions(self, commandline: str) -> List[int]:
        """Interface for :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
        subclasses to convert from a commandline invocation to a sequence of
        actions.

        See also :meth:`commandline()
        <compiler_gym.envs.CompilerEnv.commandline>`.

        Calling this method on a :class:`CompilerEnv
        <compiler_gym.envs.CompilerEnv>` instance raises
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
            benchmark=str(self.benchmark) if self.benchmark else None,
            reward=self.episode_reward,
            walltime=self.episode_walltime,
            commandline=self.commandline(),
        )

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
        return self._benchmark_in_use

    @benchmark.setter
    def benchmark(self, benchmark: Union[str, Benchmark]):
        if self.in_episode:
            warnings.warn(
                "Changing the benchmark has no effect until reset() is called"
            )
        if isinstance(benchmark, str):
            benchmark_object = self.datasets.benchmark(benchmark)
            self.logger.debug("Setting benchmark by name: %s", benchmark_object)
            self._next_benchmark = benchmark_object
        elif isinstance(benchmark, Benchmark):
            self.logger.debug("Setting benchmark: %s", benchmark.uri)
            self._next_benchmark = benchmark
        else:
            raise TypeError(
                f"Expected a Benchmark or str, received: '{type(benchmark).__name__}'"
            )

    @property
    def reward_space(self) -> Optional[Reward]:
        """The default reward space that is used to return a reward value from
        :func:`~step()`.

        :getter: Returns a :class:`Reward <compiler_gym.spaces.Reward>`,
            or :code:`None` if not set.
        :setter: Set the default reward space.
        """
        return self.reward_space_spec

    @reward_space.setter
    def reward_space(self, reward_space: Optional[Union[str, Reward]]) -> None:
        # Coerce the observation space into a string.
        reward_space: Optional[str] = (
            reward_space.id if isinstance(reward_space, Reward) else reward_space
        )

        if reward_space:
            if reward_space not in self.reward.spaces:
                raise LookupError(f"Reward space not found: {reward_space}")
            self.reward_space_spec = self.reward.spaces[reward_space]
            self.reward_range = (
                self.reward_space_spec.min,
                self.reward_space_spec.max,
            )
        else:
            # If no reward space is being used then set the reward range to
            # unbounded.
            self.reward_space_spec = None
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

        :return: A new environment instance.
        """
        if not self.in_episode:
            if self.actions and not self.in_episode:
                self.logger.warning(
                    "Parent service of fork() has died, replaying state"
                )
                self.apply(self.state)
            else:
                self.reset()

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

        # Now that we have initialized the environment with the current state,
        # set the benchmark so that calls to new_env.reset() will correctly
        # revert the environment to the initial benchmark state.
        #
        # pylint: disable=protected-access
        new_env._next_benchmark = self._benchmark_in_use

        # Set the "visible" name of the current benchmark to hide the fact that
        # we loaded from a custom bitcode file.
        new_env._benchmark_in_use = self._benchmark_in_use

        # Create copies of the mutable reward and observation spaces. This
        # is required to correctly calculate incremental updates.
        new_env.reward.spaces = deepcopy(self.reward.spaces)
        new_env.observation.spaces = deepcopy(self.observation.spaces)

        # Set the default observation and reward types. Note the use of IDs here
        # to prevent passing the spaces by reference.
        if self.observation_space:
            new_env.observation_space = self.observation_space_spec.id
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
        again.

        .. note::

            Internally, CompilerGym environments may launch subprocesses and use
            temporary files to communicate between the environment and the
            underlying compiler (see :ref:`compiler_gym.service
            <compiler_gym/service:compiler_gym.service>` for details). This
            means it is important to call :meth:`env.close()
            <compiler_gym.envs.CompilerEnv.close>` after use to free up
            resources and prevent orphan subprocesses or files. We recommend
            using the :code:`with`-statement pattern for creating environments:

                >>> with gym.make("llvm-autophase-ic-v0") as env:
                ...    env.reset()
                ...    # use env how you like

            This removes the need to call :meth:`env.close()
            <compiler_gym.envs.CompilerEnv.close>` yourself.
        """
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

        :return: The initial observation.

        :raises BenchmarkInitError: If the benchmark is invalid. In this case,
            another benchmark must be used.

        :raises TypeError: If no benchmark has been set, and the environment
            does not have a default benchmark to select from.
        """
        if not self._next_benchmark:
            raise TypeError(
                "No benchmark set. Set a benchmark using "
                "`env.reset(benchmark=benchmark)`. Use `env.datasets` to "
                "access the available benchmarks."
            )

        # Start a new service if required.
        if self.service is None:
            self.service = CompilerGymServiceConnection(
                self._service_endpoint, self._connection_settings
            )

        self.action_space_name = action_space or self.action_space_name

        # Stop an existing episode.
        if self.in_episode:
            self.service(
                self.service.stub.EndSession,
                EndSessionRequest(session_id=self._session_id),
            )
            self._session_id = None

        # Update the user requested benchmark, if provided.
        if benchmark:
            self.benchmark = benchmark
        self._benchmark_in_use = self._next_benchmark

        start_session_request = StartSessionRequest(
            benchmark=self._benchmark_in_use.uri,
            action_space=(
                [a.name for a in self.action_spaces].index(self.action_space_name)
                if self.action_space_name
                else 0
            ),
            observation_space=(
                [self.observation_space_spec.index] if self.observation_space else None
            ),
        )

        try:
            reply = self.service(self.service.stub.StartSession, start_session_request)
        except FileNotFoundError:
            # The benchmark was not found, so try adding it and repeating the
            # request.
            self.service(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(benchmark=[self._benchmark_in_use.proto]),
            )
            reply = self.service(self.service.stub.StartSession, start_session_request)
        except (ServiceError, ServiceTransportError, TimeoutError) as e:
            # Abort and retry on error.
            self.logger.warning("%s on reset(): %s", type(e).__name__, e)
            if self.service:
                self.service.close()
            self.service = None

            if retry_count >= self._connection_settings.init_max_attempts:
                raise OSError(
                    f"Failed to reset environment after {retry_count - 1} attempts.\n"
                    f"Last error ({type(e).__name__}): {e}"
                ) from e
            else:
                return self.reset(
                    benchmark=benchmark,
                    action_space=action_space,
                    retry_count=retry_count + 1,
                )

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
            if len(reply.observation) != 1:
                raise OSError(
                    f"Expected one observation from service, received {len(reply.observation)}"
                )
            return self.observation.spaces[self.observation_space_spec.id].translate(
                reply.observation[0]
            )

    def step(self, action: Union[int, Iterable[int]]) -> StepType:
        """Take a step.

        :param action: An action, or a sequence of actions. When multiple
            actions are provided the observation and reward are returned after
            running all of the actions.

        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if default observation/reward is not set. If done is
            True, observation and reward may also be None (e.g. because the
            service failed).

        :raises SessionNotFound: If :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.
        """
        if not self.in_episode:
            raise SessionNotFound("Must call reset() before step()")
        actions = action if isinstance(action, IterableType) else [action]
        observation, reward = None, None

        # Build the list of observations that must be computed by the backend
        # service to generate the user-requested observation and reward.
        # TODO(cummins): We could de-duplicate this list to improve efficiency
        # when multiple redundant copies of the same observation space are
        # requested.
        observation_indices, observation_spaces = [], []
        if self.observation_space:
            observation_indices.append(self.observation_space_spec.index)
            observation_spaces.append(self.observation_space_spec.id)
        if self.reward_space:
            observation_indices += [
                self.observation.spaces[obs].index
                for obs in self.reward_space.observation_spaces
            ]
            observation_spaces += self.reward_space.observation_spaces

        # Record the actions.
        self.actions += actions

        # Send the request to the backend service.
        request = StepRequest(
            session_id=self._session_id,
            action=actions,
            observation_space=observation_indices,
        )
        try:
            reply = _wrapped_step(self.service, request)
        except (
            ServiceError,
            ServiceTransportError,
            ServiceOSError,
            TimeoutError,
            SessionNotFound,
        ) as e:
            # Gracefully handle "expected" error types. These non-fatal errors
            # end the current episode and provide some diagnostic information to
            # the user through the `info` dict.
            self.close()
            info = {
                "error_type": type(e).__name__,
                "error_details": str(e),
            }
            if self.reward_space:
                reward = self.reward_space.reward_on_error(self.episode_reward)
            if self.observation_space:
                observation = self.observation_space_spec.default_value
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
        observation = self.observation[self.observation_space_spec.id]
        if mode == "human":
            print(observation)
        elif mode == "ansi":
            return str(observation)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    @deprecated(
        version="0.1.8",
        reason=(
            "Use :meth:`env.datasets.benchmarks() <compiler_gym.datasets.Datasets.benchmarks>` instead. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def benchmarks(self) -> Iterable[str]:
        """Enumerate a (possible unbounded) list of available benchmarks."""
        return self.datasets.benchmark_uris()

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

    @deprecated(
        version="0.1.8",
        reason=(
            "Datasets are now installed automatically, there is no need to call :code:`require()`. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def require_datasets(self, datasets: List[Union[str, Dataset]]) -> bool:
        """Deprecated function for managing datasets.

        Datasets are now installed automatically. See :class:`env.datasets
        <compiler_gym.datasets.Datasets>`.

        :param datasets: A list of datasets to require. Each dataset is the name
            of an available dataset, the URL of a dataset to download, or a
            :class:`Dataset <compiler_gym.datasets.Dataset>` instance.

        :return: :code:`True` if one or more datasets were downloaded, or
            :code:`False` if all datasets were already available.
        """
        return False

    @deprecated(
        version="0.1.8",
        reason=(
            "Use :meth:`env.datasets.require() <compiler_gym.datasets.Datasets.require>` instead. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def require_dataset(self, dataset: Union[str, Dataset]) -> bool:
        """Deprecated function for managing datasets.

        Datasets are now installed automatically. See :class:`env.datasets
        <compiler_gym.datasets.Datasets>`.

        :param dataset: The name of the dataset to download, the URL of the
            dataset, or a :class:`Dataset <compiler_gym.datasets.Dataset>`
            instance.

        :return: :code:`True` if the dataset was downloaded, or :code:`False` if
            the dataset was already available.
        """
        return False

    @deprecated(
        version="0.1.8",
        reason=(
            "Use :meth:`env.datasets.add() <compiler_gym.datasets.Datasets.require>` instead. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def register_dataset(self, dataset: Dataset) -> bool:
        """Register a new dataset.

        Example usage:

            >>> my_dataset = Dataset(name="my-dataset-v0", ...)
            >>> env = gym.make("llvm-v0")
            >>> env.register_dataset(my_dataset)
            >>> env.benchmark = "my-dataset-v0/1"

        :param dataset: A :class:`Dataset <compiler_gym.datasets.Dataset>`
            instance describing the new dataset.

        :return: :code:`True` if the dataset was added, else :code:`False`.

        :raises ValueError: If a dataset with this name is already registered.
        """
        return self.datasets.add(dataset)

    def apply(self, state: CompilerEnvState) -> None:  # noqa
        """Replay this state on the given an environment.

        :param env: A :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
            instance.

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
        _, _, done, info = self.step(actions)
        if done:
            raise ValueError(
                f"Environment terminated with error: `{info.get('error_details')}`"
            )

    def validate(self, state: Optional[CompilerEnvState] = None) -> ValidationResult:
        """Validate an environment's state.

        :param state: A state to environment. If not provided, the current state
            is validated.

        :returns: A :class:`ValidationResult <compiler_gym.ValidationResult>`.
        """
        if state:
            self.reset(benchmark=state.benchmark)
            in_place = False
        else:
            state = self.state
            in_place = True

        assert self.in_episode

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

    @deprecated(
        version="0.1.8",
        reason=(
            "Use :meth:`env.validate() "
            "<compiler_gym.datasets.Benchmark.validate>` instead. "
            "`More information <https://github.com/facebookresearch/CompilerGym/issues/45>`_."
        ),
    )
    def get_benchmark_validation_callback(
        self,
    ) -> Optional[Callable[["CompilerEnv"], Iterable[ValidationError]]]:
        """Return a callback that validates benchmark semantics, if available."""

        def composed(env):
            for validation_cb in self.benchmark.validation_callbacks():
                errors = validation_cb(env)
                if errors:
                    yield from errors

        if self.benchmark.validation_callbacks():
            return composed
