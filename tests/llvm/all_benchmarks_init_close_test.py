# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""

from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_init_benchmark(env: CompilerEnv, benchmark_name: str):
    """Create an environment for each benchmark and close it."""
    env.reset(benchmark=benchmark_name)
    assert env.benchmark == benchmark_name
    env.close()


if __name__ == "__main__":
    main()
