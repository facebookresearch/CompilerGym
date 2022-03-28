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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
from deprecated.sphinx import deprecated
from gym.spaces import Space

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.datasets import Benchmark, Dataset, Datasets
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
    ServiceOSError,
    ServiceTransportError,
    SessionNotFound,
)
from compiler_gym.service.connection import ServiceIsClosed
from compiler_gym.service.proto import AddBenchmarkRequest
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import (
    EndSessionReply,
    EndSessionRequest,
    Event,
    ForkSessionReply,
    ForkSessionRequest,
    GetVersionReply,
    GetVersionRequest,
    SendSessionParameterReply,
    SendSessionParameterRequest,
    SessionParameter,
    StartSessionRequest,
    StepReply,
    StepRequest,
    proto_to_action_space,
)
from compiler_gym.spaces import DefaultRewardFromObservation, NamedDiscrete, Reward
from compiler_gym.util.gym_type_hints import (
    ActionType,
    ObservationType,
    RewardType,
    StepType,
)
from compiler_gym.util.shell_format import plural
from compiler_gym.util.timer import Timer
from compiler_gym.validation_error import ValidationError
from compiler_gym.validation_result import ValidationResult
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView

logger = logging.getLogger(__name__)

# NOTE(cummins): This is only required to prevent a name conflict with the now
# deprecated CompilerEnv.logger attribute. This can be removed once the logger
# attribute is removed, scheduled for release 0.2.3.
_logger = logger


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

    See :code:`compiler_gym.COMPILER_GYM_ENVS` for a list of registered
    environment names.

    Alternatively, an environment can be constructed directly, such as by
    connecting to a running compiler service at :code:`localhost:8080` (see
    :doc:`this document </compiler_gym/service>` for more details):

        >>> env = CompilerEnv(
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

    :ivar service: A connection to the underlying compiler service.

    :vartype service: compiler_gym.service.CompilerGymServiceConnection

    :ivar action_spaces: A list of supported action space names.

    :vartype action_spaces: List[str]

    :ivar actions: The list of actions that have been performed since the
        previous call to :func:`reset`.

    :vartype actions: List[ActionType]

    :ivar reward_range: A tuple indicating the range of reward values. Default
        range is (-inf, +inf).

    :vartype reward_range: Tuple[float, float]

    :ivar observation: A view of the available observation spaces that permits
        on-demand computation of observations.

    :vartype observation: compiler_gym.views.ObservationView

    :ivar reward: A view of the available reward spaces that permits on-demand
        computation of rewards.

    :vartype reward: compiler_gym.views.RewardView

    :ivar episode_reward: If :func:`CompilerEnv.reward_space
        <compiler_gym.envs.CompilerGym.reward_space>` is set, this value is the
        sum of all rewards for the current episode.

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
        derived_observation_spaces: Optional[List[Dict[str, Any]]] = None,
        connection_settings: Optional[ConnectionOpts] = None,
        service_connection: Optional[CompilerGymServiceConnection] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Construct and initialize a CompilerGym environment.

        In normal use you should use :code:`gym.make(...)` rather than calling
        the constructor directly.

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

        :param derived_observation_spaces: An optional list of arguments to be
            passed to :meth:`env.observation.add_derived_space()
            <compiler_gym.views.observation.Observation.add_derived_space>`.

        :param connection_settings: The settings used to establish a connection
            with the remote service.

        :param service_connection: An existing compiler gym service connection
            to use.

        :raises FileNotFoundError: If service is a path to a file that is not
            found.

        :raises TimeoutError: If the compiler service fails to initialize within
            the parameters provided in :code:`connection_settings`.
        """
        # NOTE(cummins): Logger argument deprecated and scheduled to be removed
        # in release 0.2.3.
        if logger:
            warnings.warn(
                "The `logger` argument is deprecated on CompilerEnv.__init__() "
                "and will be removed in a future release. All CompilerEnv "
                "instances share a logger named compiler_gym.envs.compiler_env",
                DeprecationWarning,
            )

        self.metadata = {"render.modes": ["human", "ansi"]}

        # A compiler service supports multiple simultaneous environments. This
        # session ID is used to identify this environment.
        self._session_id: Optional[int] = None

        self._service_endpoint: Union[str, Path] = service
        self._connection_settings = connection_settings or ConnectionOpts()

        self.service = service_connection or CompilerGymServiceConnection(
            endpoint=self._service_endpoint,
            opts=self._connection_settings,
        )
        self.datasets = Datasets(datasets or [])

        self.action_space_name = action_space

        # If no reward space is specified, generate some from numeric observation spaces
        rewards = rewards or [
            DefaultRewardFromObservation(obs.name)
            for obs in self.service.observation_spaces
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
            # are no benchmarks available. This is to allow CompilerEnv to be
            # used without any datasets by setting a benchmark before/during the
            # first reset() call.
            pass

        # Process the available action, observation, and reward spaces.
        self.action_spaces = [
            proto_to_action_space(space) for space in self.service.action_spaces
        ]

        self.observation = self._observation_view_type(
            raw_step=self.raw_step,
            spaces=self.service.observation_spaces,
        )
        self.reward = self._reward_view_type(rewards, self.observation)

        # Register any derived observation spaces now so that the observation
        # space can be set below.
        for derived_observation_space in derived_observation_spaces or []:
            self.observation.add_derived_space_internal(**derived_observation_space)

        # Lazily evaluated version strings.
        self._versions: Optional[GetVersionReply] = None

        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None

        # Mutable state initialized in reset().
        self.reward_range: Tuple[float, float] = (-np.inf, np.inf)
        self.episode_reward: Optional[float] = None
        self.episode_start_time: float = time()
        self.actions: List[ActionType] = []

        # Initialize the default observation/reward spaces.
        self.observation_space_spec: Optional[ObservationSpaceSpec] = None
        self.reward_space_spec: Optional[Reward] = None
        self.observation_space = observation_space
        self.reward_space = reward_space

    @property
    @deprecated(
        version="0.2.1",
        reason=(
            "The `CompilerEnv.logger` attribute is deprecated. All CompilerEnv "
            "instances share a logger named compiler_gym.envs.compiler_env"
        ),
    )
    def logger(self):
        return _logger

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

    def commandline_to_actions(self, commandline: str) -> List[ActionType]:
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
    def action_space(self) -> Space:
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
    def benchmark(self, benchmark: Union[str, Benchmark, BenchmarkUri]):
        if self.in_episode:
            warnings.warn(
                "Changing the benchmark has no effect until reset() is called"
            )
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
            reward_space.name if isinstance(reward_space, Reward) else reward_space
        )

        if reward_space:
            if reward_space not in self.reward.spaces:
                raise LookupError(f"Reward space not found: {reward_space}")
            # The reward space remains unchanged, nothing to do.
            if reward_space == self.reward_space:
                return
            self.reward_space_spec = self.reward.spaces[reward_space]
            self.reward_range = (
                self.reward_space_spec.min,
                self.reward_space_spec.max,
            )
            # Reset any cumulative rewards, if we're in an episode.
            if self.in_episode:
                self.episode_reward = 0
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
    def observation_space(self) -> Optional[Space]:
        """The observation space that is used to return an observation value in
        :func:`~step()`.

        :getter: Returns the underlying observation space, or :code:`None` if
            not set.
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

    def _init_kwargs(self) -> Dict[str, Any]:
        """Retturn a dictionary of keyword arguments used to initialize the
        environment.
        """
        return {
            "action_space": self.action_space,
            "benchmark": self.benchmark,
            "connection_settings": self._connection_settings,
            "service": self._service_endpoint,
        }

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
            actions = self.actions.copy()
            self.reset()
            if actions:
                logger.warning("Parent service of fork() has died, replaying state")
                _, _, done, _ = self.multistep(actions)
                assert not done, "Failed to replay action sequence"

        request = ForkSessionRequest(session_id=self._session_id)
        try:
            reply: ForkSessionReply = self.service(
                self.service.stub.ForkSession, request
            )

            # Create a new environment that shares the connection.
            new_env = type(self)(**self._init_kwargs(), service_connection=self.service)

            # Set the session ID.
            new_env._session_id = reply.session_id  # pylint: disable=protected-access
            new_env.observation.session_id = reply.session_id

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
            except ServiceIsClosed:
                # This error can be safely ignored as it means that the service
                # is already offline.
                pass
            except Exception as e:
                logger.warning(
                    "Failed to end active compiler session on close(): %s (%s)",
                    e,
                    type(e).__name__,
                )
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

        def _retry(error) -> Optional[ObservationType]:
            """Abort and retry on error."""
            # Log the error that we are recovering from, but treat
            # ServiceIsClosed errors as unimportant since we know what causes
            # them.
            log_severity = (
                logger.debug if isinstance(error, ServiceIsClosed) else logger.warning
            )
            log_severity("%s during reset(): %s", type(error).__name__, error)

            if self.service:
                try:
                    self.service.close()
                except ServiceError as e:
                    # close() can raise ServiceError if the service exists with
                    # a non-zero return code. We swallow the error here as we
                    # are about to retry.
                    logger.debug(
                        "Ignoring service error during reset() attempt: %s (%s)",
                        e,
                        type(e).__name__,
                    )
            self.service = None

            if retry_count >= self._connection_settings.init_max_attempts:
                raise OSError(
                    "Failed to reset environment using benchmark "
                    f"{self.benchmark} after {retry_count - 1} attempts.\n"
                    f"Last error ({type(error).__name__}): {error}"
                ) from error
            else:
                return self.reset(
                    benchmark=benchmark,
                    action_space=action_space,
                    retry_count=retry_count + 1,
                )

        def _call_with_error(
            stub_method, *args, **kwargs
        ) -> Tuple[Optional[Exception], Optional[Any]]:
            """Call the given stub method. And return an <error, return> tuple."""
            try:
                return None, self.service(stub_method, *args, **kwargs)
            except (ServiceError, ServiceTransportError, TimeoutError) as e:
                return e, None

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
            logger.debug("Ending session %d", self._session_id)
            error, _ = _call_with_error(
                self.service.stub.EndSession,
                EndSessionRequest(session_id=self._session_id),
            )
            if error:
                logger.warning(
                    "Failed to stop session %d with %s: %s",
                    self._session_id,
                    type(error).__name__,
                    error,
                )
            self._session_id = None

        # Update the user requested benchmark, if provided.
        if benchmark:
            self.benchmark = benchmark
        self._benchmark_in_use = self._next_benchmark

        # When always_send_benchmark_on_reset option is enabled, the entire
        # benchmark program is sent with every StartEpisode request. Otherwise
        # only the URI of the benchmark is sent. In cases where benchmarks are
        # reused between calls to reset(), sending the URI is more efficient as
        # the service can cache the benchmark. In cases where reset() is always
        # called with a different benchmark, this causes unnecessary roundtrips
        # as every StartEpisodeRequest receives a FileNotFound response.
        if self.service.opts.always_send_benchmark_on_reset:
            self._benchmark_in_use_proto = self._benchmark_in_use.proto
        else:
            self._benchmark_in_use_proto.uri = str(self._benchmark_in_use.uri)

        start_session_request = StartSessionRequest(
            benchmark=self._benchmark_in_use_proto,
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
            error, reply = _call_with_error(
                self.service.stub.StartSession, start_session_request
            )
            if error:
                return _retry(error)
        except FileNotFoundError:
            # The benchmark was not found, so try adding it and then repeating
            # the request.
            error, _ = _call_with_error(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(benchmark=[self._benchmark_in_use.proto]),
            )
            if error:
                return _retry(error)
            error, reply = _call_with_error(
                self.service.stub.StartSession, start_session_request
            )
            if error:
                return _retry(error)

        self._session_id = reply.session_id
        self.observation.session_id = reply.session_id
        self.reward.get_cost = self.observation.__getitem__
        self.episode_start_time = time()
        self.actions = []

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = proto_to_action_space(reply.new_action_space)

        self.reward.reset(benchmark=self.benchmark, observation_view=self.observation)
        if self.reward_space:
            self.episode_reward = 0.0

        if self.observation_space:
            if len(reply.observation) != 1:
                raise OSError(
                    f"Expected one observation from service, received {len(reply.observation)}"
                )
            return self.observation.spaces[self.observation_space_spec.id].translate(
                reply.observation[0]
            )

    def raw_step(
        self,
        actions: Iterable[ActionType],
        observation_spaces: List[ObservationSpaceSpec],
        reward_spaces: List[Reward],
    ) -> StepType:
        """Take a step.

        :param actions: A list of actions to be applied.

        :param observations: A list of observations spaces to compute
            observations from. These are evaluated after the actions are
            applied.

        :param rewards: A list of reward spaces to compute rewards from. These
            are evaluated after the actions are applied.

        :return: A tuple of observations, rewards, done, and info. Observations
            and rewards are lists.

        :raises SessionNotFound: If :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.

        .. warning::

            Don't call this method directly, use :meth:`step()
            <compiler_gym.envs.CompilerEnv.step>` or :meth:`multistep()
            <compiler_gym.envs.CompilerEnv.multistep>` instead. The
            :meth:`raw_step() <compiler_gym.envs.CompilerEnv.step>` method is an
            implementation detail.
        """
        if not self.in_episode:
            raise SessionNotFound("Must call reset() before step()")

        reward_observation_spaces: List[ObservationSpaceSpec] = []
        for reward_space in reward_spaces:
            reward_observation_spaces += [
                self.observation.spaces[obs] for obs in reward_space.observation_spaces
            ]

        observations_to_compute: List[ObservationSpaceSpec] = list(
            set(observation_spaces).union(set(reward_observation_spaces))
        )
        observation_space_index_map: Dict[ObservationSpaceSpec, int] = {
            observation_space: i
            for i, observation_space in enumerate(observations_to_compute)
        }

        # Record the actions.
        self.actions += actions

        # Send the request to the backend service.
        request = StepRequest(
            session_id=self._session_id,
            action=[Event(int64_value=a) for a in actions],
            observation_space=[
                observation_space.index for observation_space in observations_to_compute
            ],
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
            info = {
                "error_type": type(e).__name__,
                "error_details": str(e),
            }

            try:
                self.close()
            except ServiceError as e:
                # close() can raise ServiceError if the service exists with a
                # non-zero return code. We swallow the error here but propagate
                # the diagnostic message.
                info[
                    "error_details"
                ] += f". Additional error during environment closing: {e}"

            default_observations = [
                observation_space.default_value
                for observation_space in observation_spaces
            ]
            default_rewards = [
                float(reward_space.reward_on_error(self.episode_reward))
                for reward_space in reward_spaces
            ]
            return default_observations, default_rewards, True, info

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = proto_to_action_space(reply.new_action_space)

        # Translate observations to python representations.
        if len(reply.observation) != len(observations_to_compute):
            raise ServiceError(
                f"Requested {len(observations_to_compute)} observations "
                f"but received {len(reply.observation)}"
            )
        computed_observations = [
            observation_space.translate(value)
            for observation_space, value in zip(
                observations_to_compute, reply.observation
            )
        ]

        # Get the user-requested observation.
        observations: List[ObservationType] = [
            computed_observations[observation_space_index_map[observation_space]]
            for observation_space in observation_spaces
        ]

        # Update and compute the rewards.
        rewards: List[RewardType] = []
        for reward_space in reward_spaces:
            reward_observations = [
                computed_observations[
                    observation_space_index_map[
                        self.observation.spaces[observation_space]
                    ]
                ]
                for observation_space in reward_space.observation_spaces
            ]
            rewards.append(
                float(
                    reward_space.update(actions, reward_observations, self.observation)
                )
            )

        info = {
            "action_had_no_effect": reply.action_had_no_effect,
            "new_action_space": reply.HasField("new_action_space"),
        }

        return observations, rewards, reply.end_of_session, info

    def step(  # pylint: disable=arguments-differ
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

        :raises SessionNotFound: If :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.
        """
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
                "Argument `observations` of CompilerEnv.step has been "
                "renamed `observation_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            observation_spaces = observations
        if rewards is not None:
            warnings.warn(
                "Argument `rewards` of CompilerEnv.step has been renamed "
                "`reward_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            reward_spaces = rewards
        return self.multistep([action], observation_spaces, reward_spaces)

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

        :raises SessionNotFound: If :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>` has not been called.
        """
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
        else:
            observation_spaces_to_compute: List[ObservationSpaceSpec] = []

        # Coerce reward spaces into a list of Reward instances.
        if reward_spaces:
            reward_spaces_to_compute: List[Reward] = [
                rew if isinstance(rew, Reward) else self.reward.spaces[rew]
                for rew in reward_spaces
            ]
        elif self.reward_space:
            reward_spaces_to_compute: List[Reward] = [self.reward_space]
        else:
            reward_spaces_to_compute: List[Reward] = []

        # Perform the underlying environment step.
        observation_values, reward_values, done, info = self.raw_step(
            actions, observation_spaces_to_compute, reward_spaces_to_compute
        )

        # Translate observations lists back to the appropriate types.
        if observation_spaces is None and self.observation_space_spec:
            observation_values = observation_values[0]
        elif not observation_spaces_to_compute:
            observation_values = None

        # Translate reward lists back to the appropriate types.
        if reward_spaces is None and self.reward_space:
            reward_values = reward_values[0]
            # Update the cumulative episode reward
            self.episode_reward += reward_values
        elif not reward_spaces_to_compute:
            reward_values = None

        return observation_values, reward_values, done, info

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
        """Replay this state on the given an environment.

        :param env: A :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
            instance.

        :raises ValueError: If this state cannot be applied.
        """
        if not self.in_episode:
            self.reset(benchmark=state.benchmark)

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

        actions = self.commandline_to_actions(state.commandline)
        done = False
        for action in actions:
            _, _, done, info = self.step(action)
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

    def send_param(self, key: str, value: str) -> str:
        """Send a single <key, value> parameter to the compiler service.

        See :meth:`send_params() <compiler_gym.envs.CompilerEnv.send_params>`
        for more information.

        :param key: The parameter key.

        :param value: The parameter value.

        :return: The response from the compiler service.

        :raises SessionNotFound: If called before :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>`.
        """
        return self.send_params((key, value))[0]

    def send_params(self, *params: Iterable[Tuple[str, str]]) -> List[str]:
        """Send a list of <key, value> parameters to the compiler service.

        This provides a mechanism to send messages to the backend compilation
        session in a way that doesn't conform to the normal communication
        pattern. This can be useful for things like configuring runtime
        debugging settings, or applying "meta actions" to the compiler that are
        not exposed in the compiler's action space. Consult the documentation
        for a specific compiler service to see what parameters, if any, are
        supported.

        Must have called :meth:`reset() <compiler_gym.envs.CompilerEnv.reset>`
        first.

        :param params: A list of parameters, where each parameter is a
            :code:`(key, value)` tuple.

        :return: A list of string responses, one per parameter.

        :raises SessionNotFound: If called before :meth:`reset()
            <compiler_gym.envs.CompilerEnv.reset>`.
        """
        if not self.in_episode:
            raise SessionNotFound("Must call reset() before send_params()")

        request = SendSessionParameterRequest(
            session_id=self._session_id,
            parameter=[SessionParameter(key=k, value=v) for (k, v) in params],
        )
        reply: SendSessionParameterReply = self.service(
            self.service.stub.SendSessionParameter, request
        )
        if len(params) != len(reply.reply):
            raise OSError(
                f"Sent {len(params)} {plural(len(params), 'parameter', 'parameters')} but received "
                f"{len(reply.reply)} {plural(len(reply.reply), 'response', 'responses')} from the "
                "service"
            )

        return list(reply.reply)

    def __copy__(self) -> "CompilerEnv":
        raise TypeError(
            "CompilerEnv instances do not support shallow copies. Use deepcopy()"
        )

    def __deepcopy__(self, memo) -> "CompilerEnv":
        del memo  # unused
        return self.fork()
