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

import CGRA

import gym

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register

EXAMPLE_PY_SERVICE_BINARY: Path = Path(
    "CompileCGRA.py"
)
assert EXAMPLE_PY_SERVICE_BINARY.is_file(), "Service script not found"


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
    iteration = 0
    with gym.make("example-v0") as env:
        env.reset()
        done = False
        while not done:
            # Not 100% sure why this needs to go in an array, but it seems
            # to complain about dimensionality errors due to an issue
            # in compiler_gym --- maybe once there is a 
            action = env.action_space.sample()
            print("Starting Iteration " + str(iteration))
            print ("Action is:")
            print(action)
            observation, reward, done, info = env.step(action, observation_spaces=["ir", "CurrentInstruction", "CurrentInstructionIndex", "II"], reward_spaces=["II"])
            print ("Got observation")
            print (observation)
            print ("Got reward")
            print (reward)
            if done:
                env.reset()
                print ("Overall reward is ", reward)
            iteration += 1


if __name__ == "__main__":
    main()
