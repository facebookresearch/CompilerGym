# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import (
    CycleOverBenchmarks,
    IterateOverBenchmarks,
    RandomOrderBenchmarks,
)
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_iterate_over_benchmarks(env: LlvmEnv):
    env = IterateOverBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/dijkstra",
        ],
    )

    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/qsort"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"

    with pytest.raises(StopIteration):
        env.reset()


def test_cycle_over_benchmarks(env: LlvmEnv):
    env = CycleOverBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
        ],
    )

    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/qsort"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/qsort"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_random_order_benchmarks(env: LlvmEnv):
    env = RandomOrderBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
        ],
    )
    env.reset()
    assert env.benchmark in {
        "benchmark://cbench-v1/crc32",
        "benchmark://cbench-v1/qsort",
    }
    env.reset()
    assert env.benchmark in {
        "benchmark://cbench-v1/crc32",
        "benchmark://cbench-v1/qsort",
    }
    env.reset()
    assert env.benchmark in {
        "benchmark://cbench-v1/crc32",
        "benchmark://cbench-v1/qsort",
    }


if __name__ == "__main__":
    main()
