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


def test_iterate_over_benchmarks_fork(env: LlvmEnv):
    """Test that fork() copies over benchmark iterator state."""
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

    fkd = env.fork()
    try:
        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/qsort"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/qsort"

        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/dijkstra"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/dijkstra"
    finally:
        fkd.close()


def test_iterate_over_benchmarks_fork_shared_iterator(env: LlvmEnv):
    """Test fork() using a single benchmark iterator shared between forks."""
    env = IterateOverBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/dijkstra",
        ],
        fork_shares_iterator=True,
    )

    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"

    fkd = env.fork()
    try:
        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/qsort"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/dijkstra"
    finally:
        fkd.close()


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


def test_cycle_over_benchmarks_fork(env: LlvmEnv):
    env = CycleOverBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
        ],
    )

    fkd = env.fork()
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    try:
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/crc32"

        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/qsort"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/qsort"

        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/crc32"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/crc32"
    finally:
        fkd.close()


def test_cycle_over_benchmarks_fork_shared_iterator(env: LlvmEnv):
    env = CycleOverBenchmarks(
        env=env,
        benchmarks=[
            "benchmark://cbench-v1/crc32",
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/dijkstra",
        ],
        fork_shares_iterator=True,
    )

    fkd = env.fork()
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    try:
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/qsort"
        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/dijkstra"
        fkd.reset()
        assert fkd.benchmark == "benchmark://cbench-v1/crc32"
    finally:
        fkd.close()


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


def test_random_order_benchmarks_fork(env: LlvmEnv):
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
    fkd = env.fork()
    try:
        fkd.reset()
        env.reset()
    finally:
        fkd.close()


if __name__ == "__main__":
    main()
