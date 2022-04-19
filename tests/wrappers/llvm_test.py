# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.wrappers.llvm."""
import numpy as np
import pytest
from flaky import flaky

from compiler_gym.datasets.benchmark import BenchmarkInitError
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import RuntimePointEstimateReward
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_invalid_runtime_count(env: LlvmEnv):
    env = RuntimePointEstimateReward(env, runtime_count=-10)
    with pytest.raises(
        ValueError, match="runtimes_per_observation_count must be >= 1. Received: -10"
    ):
        env.reset()


def test_invalid_warmup_count(env: LlvmEnv):
    env = RuntimePointEstimateReward(env, warmup_count=-10)
    with pytest.raises(
        ValueError,
        match="warmup_runs_count_per_runtime_observation must be >= 0. Received: -10",
    ):
        env.reset()


def test_reward_range(env: LlvmEnv):
    env = RuntimePointEstimateReward(env, runtime_count=3)
    assert env.reward_range == (-float("inf"), float("inf"))


def test_reward_range_not_runnable_benchmark(env: LlvmEnv):
    env = RuntimePointEstimateReward(env, runtime_count=3)

    with pytest.raises(
        BenchmarkInitError, match=r"^Benchmark is not runnable: benchmark://npb-v0/1$"
    ):
        env.reset(benchmark="benchmark://npb-v0/1")


@flaky  # Runtime can fail
def test_fork(env: LlvmEnv):
    env = RuntimePointEstimateReward(env)
    with env.fork() as fkd:
        assert fkd.reward_space_spec.name == "runtime"


@pytest.mark.parametrize("runtime_count", [1, 3, 5])
@pytest.mark.parametrize("warmup_count", [0, 1, 3])
@pytest.mark.parametrize("estimator", [np.median, min])
@flaky  # Runtime can fail
def test_reward_values(env: LlvmEnv, runtime_count, warmup_count, estimator):
    env = RuntimePointEstimateReward(
        env, runtime_count=runtime_count, warmup_count=warmup_count, estimator=estimator
    )
    env.reset()

    assert env.reward_space_spec.runtime_count == runtime_count
    assert env.reward_space_spec.warmup_count == warmup_count
    assert env.reward_space_spec.estimator == estimator

    _, reward_a, done, info = env.step(env.action_space.sample())
    assert not done, info

    _, reward_b, done, info = env.step(env.action_space.sample())
    assert not done, info

    _, reward_c, done, info = env.step(env.action_space.sample())
    assert not done, info

    assert env.episode_reward == reward_a + reward_b + reward_c
    assert reward_a or reward_b or reward_c


if __name__ == "__main__":
    main()
