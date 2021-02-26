# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/envs."""
import gym

from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_benchmark_constructor_arg(env: CompilerEnv):
    env.close()  # Fixture only required to pull in dataset.

    env = gym.make("llvm-v0", benchmark="cbench-v1/dijkstra")
    try:
        assert env.benchmark == "benchmark://cbench-v1/dijkstra"
    finally:
        env.close()


if __name__ == "__main__":
    main()
