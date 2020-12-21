# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the OpenAI gym interface for compilers."""
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fasteners
import gym
import numpy as np
from gym.spaces import Space

from compiler_gym.datasets import Dataset, require
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
    observation2py,
    observation_t,
)
from compiler_gym.service.connection import ServiceTransportError
from compiler_gym.service.proto import (
    ActionRequest,
    AddBenchmarkRequest,
    Benchmark,
    EndEpisodeRequest,
    GetBenchmarksRequest,
    StartEpisodeRequest,
)
from compiler_gym.spaces import NamedDiscrete
from compiler_gym.views import ObservationView, RewardView

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
        eager_observation_space="features",
        eager_reward_space="runtime"
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
    """

    def __init__(
        self,
        service: Union[str, Path],
        benchmark: Optional[Union[str, Benchmark]] = None,
        eager_observation_space: Optional[str] = None,
        eager_reward_space: Optional[str] = None,
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
        :param eager_observation_space: Compute and return observations at each
            :func:`step()` from this space. If not provided, :func:`step()`
            returns :code:`None` for the observation value.
        :param eager_reward_space: Compute and return reward at each
            :func:`step()` from this space. If not provided, :func:`step()`
            returns :code:`None` for the reward value.
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
        self._benchmark = None
        self._user_specified_benchmark = None
        self.benchmark = benchmark

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

        # A compiler service supports multiple simultaneous environments. This
        # session ID is used to identify this environment.
        self._session_id: Optional[int] = None

        # Mutable state initialized in reset().
        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None
        self.reward_range: Tuple[float, float] = (-np.inf, np.inf)

        # Initialize eager observation/reward.
        self.eager_observation_space = eager_observation_space
        self.eager_reward_space = eager_reward_space

    def commandline(self) -> str:
        """Return the current state as a commandline invocation.

        :return: A string commandline invocation.
        """
        return ""

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
        """The name of the benchmark that is being used.

        :getter: Get the name of the current benchmark. Return :code:`None` if
            :func:`__init__` was not provided a benchmark and :func:`reset` has
            not yet been called.
        :setter: Set the benchmark to use. If :code:`None`, a random benchmark
            is selected by the service on each call to :func:`reset`.

        .. note::
            Setting a new benchmark has no effect until :func:`~reset()` is
            called on the environment.
        """
        return self._benchmark

    @benchmark.setter
    def benchmark(self, benchmark: Optional[Union[str, Benchmark]]):
        if isinstance(benchmark, Benchmark):
            self._benchmark = benchmark.uri
        else:
            self._benchmark = benchmark
        self._user_specified_benchmark = benchmark

    @property
    def eager_reward_space(self) -> Optional[str]:
        """The eager reward space. This is the reward that is returned by
        :func:`~step()`.

        :getter: Returns the name of the eager reward space, or :code:`None` if
            not set.
        :setter: Set the name of the eager reward space.

        .. note::
            Setting a new eager reward space has no effect until
            :func:`~reset()` is called on the environment.
        """
        return self._eager_reward_space or None

    @eager_reward_space.setter
    def eager_reward_space(self, eager_reward_space: Optional[str]) -> None:
        if (
            eager_reward_space is not None
            and eager_reward_space not in self.reward.ranges
        ):
            raise LookupError(f"Reward space not found: {eager_reward_space}")
        if self.in_episode:
            warnings.warn(
                "Changing eager reward space has no effect until reset() is called."
            )
        self._eager_reward = eager_reward_space is not None
        self._eager_reward_space = eager_reward_space or ""
        if self._eager_reward:
            self.reward_range = self.reward.ranges[self._eager_reward_space]
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
    def eager_observation_space(self) -> Optional[str]:
        """The eager observation space. This is the observation value that is
        returned by :func:`~step()`.

        :getter: Returns the name of the eager observation space, or
            :code:`None` if not set.
        :setter: Set the name of the eager observation space.

        .. note::
            Setting a new eager observation space has no effect until
            :func:`~reset()` is called on the environment.
        """
        return self._eager_observation_space or None

    @eager_observation_space.setter
    def eager_observation_space(self, eager_observation_space: Optional[str]) -> None:
        if (
            eager_observation_space is not None
            and eager_observation_space not in self.observation.spaces
        ):
            raise LookupError(f"Observation space not found: {eager_observation_space}")
        if self.in_episode:
            warnings.warn(
                "Changing eager observation space has no effect until reset() is called."
            )
        self._eager_observation = eager_observation_space is not None
        self._eager_observation_space = eager_observation_space or ""
        if self._eager_observation:
            self.observation_space = self.observation.spaces[
                self._eager_observation_space
            ]

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
        """
        if retry_count > self.connection_settings.init_max_attempts:
            raise OSError(f"Failed to reset environment after {retry_count} attempts")

        # Start a new service if required.
        if self.service is None:
            self.service = CompilerGymServiceConnection(
                self.service_endpoint, self.connection_settings
            )

        self.action_space_name = action_space or self.action_space_name

        # Stop an existing episode.
        if self.in_episode:
            self.service(
                self.service.stub.EndEpisode,
                EndEpisodeRequest(session_id=self._session_id),
            )

        # Add the new benchmark, if required.
        self._user_specified_benchmark = benchmark or self._user_specified_benchmark
        if isinstance(self._user_specified_benchmark, Benchmark):
            self.service(
                self.service.stub.AddBenchmark,
                AddBenchmarkRequest(benchmark=[self._user_specified_benchmark]),
            )
            self._user_specified_benchmark = self._user_specified_benchmark.uri

        try:
            reply = self.service(
                self.service.stub.StartEpisode,
                StartEpisodeRequest(
                    benchmark=self._user_specified_benchmark,
                    action_space=(
                        [a.name for a in self.action_spaces].index(
                            self.action_space_name
                        )
                        if self.action_space_name
                        else 0
                    ),
                    use_eager_observation_space=self._eager_observation,
                    eager_observation_space=(
                        self.observation.indices[self.eager_observation_space]
                        if self._eager_observation
                        else None
                    ),
                    use_eager_reward_space=self._eager_reward,
                    eager_reward_space=(
                        self.reward.indices[self.eager_reward_space]
                        if self._eager_reward
                        else None
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

        self._benchmark = reply.benchmark
        self._session_id = reply.session_id
        self.observation.session_id = reply.session_id
        self.reward.session_id = reply.session_id

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = self._make_action_space(
                self.action_space.name, reply.new_action_space.action
            )

        if self._eager_observation:
            return self.observation[self.eager_observation_space]

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
            return observation, reward, True, info

        # If the action space has changed, update it.
        if reply.HasField("new_action_space"):
            self.action_space = self._make_action_space(
                self.action_space.name, reply.action_space.action
            )

        if self._eager_observation:
            observation = self.observation.translate(
                self.eager_observation_space,
                observation2py(self.eager_observation_space, reply.observation),
            )
        if self._eager_reward:
            reward = reply.reward.reward

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
        if not self.eager_observation_space:
            raise ValueError(
                "Cannot call render() when no eager observation space is used"
            )
        observation = self.observation[self.eager_observation_space]
        if mode == "human":
            print(observation)
        elif mode == "ansi":
            return str(observation)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @property
    def benchmarks(self) -> List[str]:
        """The list of available benchmarks."""
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
