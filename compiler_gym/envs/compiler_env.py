# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
import csv
import os
import warnings
from io import StringIO
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import fasteners
import gym
import numpy as np
from gym.spaces import Space

from compiler_gym.datasets.dataset import Dataset, require
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
    observation_t,
)
from compiler_gym.service.connection import ServiceTransportError
from compiler_gym.service.proto import (
    ActionRequest,
    AddBenchmarkRequest,
    Benchmark,
    EndEpisodeRequest,
    GetBenchmarksRequest,
    GetVersionReply,
    GetVersionRequest,
    StartEpisodeRequest,
)
from compiler_gym.spaces import NamedDiscrete
from compiler_gym.views import (
    ObservationSpaceSpec,
    ObservationView,
    RewardSpaceSpec,
    RewardView,
)

# Type hints.
info_t = Dict[str, Any]
step_t = Tuple[Optional[observation_t], Optional[float], bool, info_t]


def _to_csv(*columns) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    return buf.getvalue().rstrip()


class CompilerEnvState(NamedTuple):
    """The representation of a compiler environment state.

    The state of an environment is defined as a benchmark and a sequence of
    actions that has been applied to it. For a given environment, the state
    contains the information required to reproduce the result.
    """

    benchmark: str
    """The name of the benchmark used for this episode."""

    commandline: str
    """The list of actions that produced this state, as a commandline."""

    walltime: float
    """The walltime of the episode."""

    reward: Optional[float] = None
    """The cumulative reward for this episode."""

    @staticmethod
    def csv_header() -> str:
        """Return the header string for the CSV-format.

        :return: A comma-separated string.
        """
        return _to_csv("benchmark", "reward", "walltime", "commandline")

    def to_csv(self) -> str:
        """Serialize a state to a comma separated list of values.

        :return: A comma-separated string.
        """
        return _to_csv(self.benchmark, self.reward, self.walltime, self.commandline)

    @classmethod
    def from_csv(cls, csv_string: str) -> "CompilerEnvState":
        """Construct a state from a comma separated list of values."""
        reader = csv.reader(StringIO(csv_string))
        for line in reader:
            try:
                benchmark, reward, walltime, commandline = line
                break
            except ValueError as e:
                raise ValueError(f"Failed to parse input: `{csv_string}`: {e}") from e
        else:
            raise ValueError(f"Failed to parse input: `{csv_string}`")
        return cls(
            benchmark=benchmark,
            reward=None if reward == "" else float(reward),
            walltime=float(walltime),
            commandline=commandline,
        )

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, CompilerEnvState):
            return False
        epsilon = 1e-5
        # Note that walltime is excluded from equivalence checks as two states
        # are equivalent if they define the same point in the optimization space
        # irrespective of how long it took to get there.
        return (
            self.benchmark == rhs.benchmark
            and abs(self.reward - rhs.reward) < epsilon
            and self.commandline == rhs.commandline
        )


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
        reward_space="runtime"
    )

    Once constructed, an environment can be used in exactly the same way as a
    regular :code:`gym.Env`, e.g.

    >>> observation = env.reset()
    >>> for i in range(100):
    >>>     action = env.action_space.sample()
    >>>     observation, reward, done, info = env.step(action)
    >>>     if done:
    >>>         break
    >>> print(f"Reward after {i} steps: {reward}")
    Reward after 100 steps: -0.32123

    :ivar service: A connection to the underlying compiler service.
    :vartype service: compiler_gym.service.CompilerGymServiceConnection

    :ivar action_spaces: A list of supported action space names.
    :vartype action_spaces: List[str]

    :ivar reward_range: A tuple indicating the range of reward values.
        Default range is (-inf, +inf).
    :vartype reward_range: Tuple[float, float]

    :ivar observation_space: The observation space. If eager observations are
        not set, this is :code:`None`, and :func:`step()` will return
        :code:`None` for the observation value.
    :vartype observation_space: Optional[Space]

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
    """

    def __init__(
        self,
        service: Union[str, Path],
        benchmark: Optional[Union[str, Benchmark]] = None,
        observation_space: Optional[str] = None,
        reward_space: Optional[str] = None,
        action_space: Optional[str] = None,
        connection_settings: Optional[ConnectionOpts] = None,
    ):
        """Construct and initialize a CompilerGym service environment.

        :param service: The hostname and port of a service that implements the
            CompilerGym service interface, or the path of a binary file
            which provides the CompilerGym service interface when executed.
            See :doc:`/compiler_gym/service` for details.
        :param benchmark: The name of the benchmark to use for this environment.
            The choice of benchmark can be deferred by not providing this
            argument and instead passing by choosing from the
            :code:`CompilerEnv.benchmarks` attribute and passing it to
            :func:`reset()` when called.
        :param observation_space: Compute and return observations at each
            :func:`step()` from this space. If not provided, :func:`step()`
            returns :code:`None` for the observation value.
        :param reward_space: Compute and return reward at each :func:`step()`
            from this space. If not provided, :func:`step()` returns
            :code:`None` for the reward value.
        :param action_space: The name of the action space to use. If not
            specified, the default action space for this compiler is used.
        :raises FileNotFoundError: If service is a path to a file that is not
            found.
        :raises TimeoutError: If the compiler service fails to initialize
            within the parameters provided in :code:`connection_settings`.
        """
        self.metadata = {"render.modes": ["human", "ansi"]}

        self.service_endpoint = service
        self.connection_settings = connection_settings or ConnectionOpts()
        self.datasets_site_path: Optional[Path] = None
        self.available_datasets: Dict[str, Dataset] = {}

        # The benchmark that is currently being used, and the benchmark that
        # the user requested. Those do not always correlate, since the user
        # could request a random benchmark.
        self._benchmark_in_use_uri: Optional[str] = None
        self._user_specified_benchmark_uri: Optional[str] = None
        # A map from benchmark URIs to Benchmark messages. We keep track of any
        # user-provided custom benchmarks so that we can register them with a
        # reset service.
        self._custom_benchmarks: Dict[str, Benchmark] = {}

        self.action_space_name = action_space

        self.service = CompilerGymServiceConnection(
            self.service_endpoint, self.connection_settings
        )

        # Process the available action, observation, and reward spaces.
        self.action_spaces = [
            self._make_action_space(space.name, space.action)
            for space in self.service.action_spaces
        ]
        self.observation = self._observation_view_type(
            get_observation=lambda req: self.service(
                self.service.stub.GetObservation, req
            ),
            spaces=self.service.observation_spaces,
        )
        self.reward = self._reward_view_type(
            get_reward=lambda req: self.service(self.service.stub.GetReward, req),
            spaces=self.service.reward_spaces,
        )

        # Lazily evaluated version strings.
        self._versions: Optional[GetVersionReply] = None

        # A compiler service supports multiple simultaneous environments. This
        # session ID is used to identify this environment.
        self._session_id: Optional[int] = None

        # Mutable state initialized in reset().
        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None
        self.reward_range: Tuple[float, float] = (-np.inf, np.inf)
        self.episode_reward: Optional[float] = None
        self.episode_start_time: float = time()

        # Initialize eager observation/reward and benchmark.
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.benchmark = benchmark

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
        """Return the current state as a commandline invocation.

        :return: A string commandline invocation.
        """
        return ""

    @property
    def episode_walltime(self) -> float:
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
            # Register the custom benchmark, and record the Benchmark object
            # in case of environment restart.
            self._custom_benchmarks[benchmark.uri] = benchmark
            self.service(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(benchmark=[benchmark]),
            )
        else:
            raise TypeError(f"Unsupported benchmark type: {type(benchmark).__name__}")

    @property
    def reward_space(self) -> Optional[RewardSpaceSpec]:
        """The eager reward space. This is the reward that is returned by
        :func:`~step()`.

        :getter: Returns a :class:`RewardSpaceSpec <compiler_gym.views.RewardSpaceSpec>`,
            or :code:`None` if not set.
        :setter: Set the eager reward space.

        .. note::
            Setting a new eager reward space has no effect until
            :func:`~reset()` is called on the environment.
        """
        return (
            self.reward.spaces[self._eager_reward_space]
            if self._eager_reward_space
            else None
        )

    @reward_space.setter
    def reward_space(self, reward_space: Optional[str]) -> None:
        if reward_space is not None and reward_space not in self.reward.spaces:
            raise LookupError(f"Reward space not found: {reward_space}")
        if self.in_episode:
            warnings.warn(
                "Changing eager reward space has no effect until reset() is called."
            )
        self._eager_reward: bool = reward_space is not None
        self._eager_reward_space: str = reward_space or ""
        if self._eager_reward:
            self.reward_range = self.reward.spaces[reward_space].range
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
        """The eager observation space. This is the observation value that is
        returned by :func:`~step()`.

        :getter: Returns the specification of the eager observation space, or
            :code:`None` if not set.
        :setter: Set the eager observation space.

        .. note::
            Setting a new eager observation space has no effect until
            :func:`~reset()` is called on the environment.
        """
        return self._eager_observation_space

    @observation_space.setter
    def observation_space(self, observation_space: Optional[str]) -> None:
        if (
            observation_space is not None
            and observation_space not in self.observation.spaces
        ):
            raise LookupError(f"Observation space not found: {observation_space}")
        if self.in_episode:
            warnings.warn(
                "Changing eager observation space has no effect until reset() is called."
            )
        self._eager_observation = observation_space is not None
        if self._eager_observation:
            self._eager_observation_space = self.observation.spaces[observation_space]
        else:
            self._eager_observation_space = None

    def close(self):
        """Close the environment.

        Once closed, :func:`reset` must be called before the environment is used
        again."""
        # Try and close out the episode, but errors are okay.
        if self.in_episode:
            try:
                self.service(
                    self.service.stub.EndEpisode,
                    EndEpisodeRequest(session_id=self._session_id),
                )
            except:
                pass
            self._session_id = None

        if self.service:
            self.service.close()

        self.service = None

    def __del__(self):
        # Don't let the service be orphaned if user forgot to close(), or
        # if an exception was thrown. The conditional guard is because this
        # may be called in case of early error.
        if hasattr(self, "service") and getattr(self, "service"):
            self.close()

    def reset(
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
        if retry_count > self.connection_settings.init_max_attempts:
            raise OSError(f"Failed to reset environment after {retry_count} attempts")

        # Start a new service if required.
        if self.service is None:
            self.service = CompilerGymServiceConnection(
                self.service_endpoint, self.connection_settings
            )
            # Re-register any custom benchmarks.
            self.service(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(benchmark=list(self._custom_benchmarks.values())),
            )

        self.action_space_name = action_space or self.action_space_name

        # Stop an existing episode.
        if self.in_episode:
            self.service(
                self.service.stub.EndEpisode,
                EndEpisodeRequest(session_id=self._session_id),
            )
            self._session_id = None

        # Update the user requested benchmark, if provided. NOTE: This means
        # that env.reset(benchmark=None) does NOT unset a forced benchmark.
        if benchmark:
            self.benchmark = benchmark

        try:
            reply = self.service(
                self.service.stub.StartEpisode,
                StartEpisodeRequest(
                    benchmark=self._user_specified_benchmark_uri,
                    action_space=(
                        [a.name for a in self.action_spaces].index(
                            self.action_space_name
                        )
                        if self.action_space_name
                        else 0
                    ),
                    use_eager_observation_space=self._eager_observation,
                    eager_observation_space=(
                        self.observation_space.index if self.observation_space else None
                    ),
                    use_eager_reward_space=bool(self.reward_space),
                    eager_reward_space=(
                        self.reward_space.index if self.reward_space else None
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
        self.reward.session_id = reply.session_id
        self.episode_start_time = time()

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = self._make_action_space(
                self.action_space.name, reply.new_action_space.action
            )

        if self.reward_space:
            self.episode_reward = 0

        if self._eager_observation:
            return self.observation[self.observation_space.id]

    def step(self, action: int) -> step_t:
        """Take a step.

        :param action: Value from the action_space.
        :return: A tuple of observation, reward, done, and info. Observation and
            reward are None if eager observation/reward is not set. If done
            is True, observation and reward may also be None (e.g. because the
            service failed).
        """
        assert self.in_episode, "Must call reset() before step()"
        observation, reward = None, None
        request = ActionRequest(session_id=self._session_id, action=[action])
        try:
            reply = self.service(self.service.stub.TakeAction, request)
        except (ServiceError, ServiceTransportError, TimeoutError) as e:
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

        if self.observation_space:
            observation = self.observation_space.cb(reply.observation)
        if self._eager_reward:
            reward = reply.reward.reward
            self.episode_reward += reward

        info = {
            "action_had_no_effect": reply.action_had_no_effect,
            "new_action_space": reply.HasField("new_action_space"),
        }

        return observation, reward, reply.end_of_episode, info

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
        :raises TypeError: If eager observations are not set, or if the
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

    def register_dataset(self, dataset: Dataset) -> None:
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
        :raises ValueError: If a dataset with this name is already registered.
        """
        if dataset.name in self.available_datasets:
            raise ValueError(f"Dataset already registered with name: {dataset.name}")
        self.available_datasets[dataset.name] = dataset
