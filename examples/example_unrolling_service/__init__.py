# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
import os
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

UNROLLING_PY_SERVICE_BINARY: Path = runfiles_path(
    "examples/example_unrolling_service/service_py/example-unrolling-service-py"
)


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_runtime = 1

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_runtime = observation_view["runtime"]

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.baseline_runtime is None:
            self.baseline_runtime = observations[0]

        # Here we are using Contextual Bandits: the number of steps the RL agent has to take before
        # the environment terminates is one. In Contextual Bandits the learner tries
        # to find a single best action in the current state. It involves learning to search for best actions and trial-and-error
        reward = float(self.baseline_runtime - observations[0]) / self.baseline_runtime
        return reward


class SizeReward(Reward):
    """An example reward that uses changes in the "size" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="size",
            observation_spaces=["size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.baseline_size = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.baseline_size = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.baseline_size is None:
            self.baseline_size = observations[0]

        # Here we are using Contextual Bandits: the number of steps the RL agent has to take before
        # the environment terminates is one. In Contextual Bandits the learner tries
        # to find a single best action in the current state. It involves learning to search for best actions and trial-and-error
        reward = float(self.baseline_size - observations[0]) / self.baseline_size
        return reward


class UnrollingDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://unrolling-v0",
            license="MIT",
            description="Unrolling example dataset",
            site_data_base=site_data_path(
                "example_dataset"
            ),  # TODO: what should we set this to? we are not using it
        )
        benchmarks_dir_path = os.path.join(os.path.dirname(__file__), "benchmarks")
        self._benchmarks = {
            "benchmark://unrolling-v0/offsets1": Benchmark.from_file(
                "benchmark://unrolling-v0/offsets1",
                os.path.join(benchmarks_dir_path, "offsets1.c"),
            ),
            "benchmark://unrolling-v0/conv2d": Benchmark.from_file(
                "benchmark://unrolling-v0/conv2d",
                os.path.join(benchmarks_dir_path, "conv2d.c"),
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from self._benchmarks.keys()

    def benchmark(self, uri: str) -> Benchmark:
        if uri in self._benchmarks:
            return self._benchmarks[uri]
        else:
            raise LookupError("Unknown program name")


# Register the unrolling example service on module import. After importing this module,
# the unrolling-py-v0 environment will be available to gym.make(...).

register(
    id="unrolling-py-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": UNROLLING_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward(), SizeReward()],
        "datasets": [UnrollingDataset()],
    },
)
