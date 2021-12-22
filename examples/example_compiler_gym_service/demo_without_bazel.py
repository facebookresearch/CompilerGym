# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python example_compiler_gym_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
from pathlib import Path
from typing import Iterable

import gym

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register

EXAMPLE_PY_SERVICE_BINARY: Path = Path(
    "example_compiler_gym_service/service_py/example_service.py"
)
assert EXAMPLE_PY_SERVICE_BINARY.is_file(), "Service script not found"


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
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]

        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


class ExampleDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://example-v0",
            license="MIT",
            description="An example dataset",
        )
        self._benchmarks = {
            "/foo": Benchmark.from_file_contents(
                "benchmark://example-v0/foo", "Ir data".encode("utf-8")
            ),
            "/bar": Benchmark.from_file_contents(
                "benchmark://example-v0/bar", "Ir data".encode("utf-8")
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from (f"benchmark://example-v0{k}" for k in self._benchmarks.keys())

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown program name")


# Register the environment for use with gym.make(...).
register(
    id="example-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": EXAMPLE_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
        "datasets": [ExampleDataset()],
    },
)


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.DEBUG)

    # Create the environment using the regular gym.make(...) interface.
    with gym.make("example-v0") as env:
        env.reset()
        for _ in range(20):
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()


if __name__ == "__main__":
    main()
