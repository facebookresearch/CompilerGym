# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the AnghaBench dataset."""
import sys
from itertools import islice

import gym
import numpy as np
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.datasets import BenchmarkInitError
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import LlvmStressDataset
from tests.pytest_plugins.common import is_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def llvm_stress_dataset() -> LlvmStressDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["generator://llvm-stress-v0"]
    finally:
        env.close()
    yield ds


def test_llvm_stress_size(llvm_stress_dataset: LlvmStressDataset):
    assert llvm_stress_dataset.size == float("inf")


@pytest.mark.parametrize("index", range(3) if is_ci() else range(250))
def test_llvm_stress_random_select(
    env: LlvmEnv, llvm_stress_dataset: LlvmStressDataset, index: int
):
    env.observation_space = "InstCountDict"

    uri = next(islice(llvm_stress_dataset.benchmark_uris(), index, None))
    benchmark = llvm_stress_dataset.benchmark(uri)

    # As of the current version (LLVM 10.0.0), programs generated with the
    # following seeds emit an error when compiled: "Cannot emit physreg copy
    # instruction".
    FAILING_SEEDS = {"linux": {173, 239}, "darwin": {173}}[sys.platform]

    if index in FAILING_SEEDS:
        with pytest.raises(
            BenchmarkInitError, match="Cannot emit physreg copy instruction"
        ):
            env.reset(benchmark=benchmark)
    else:
        instcount = env.reset(benchmark=benchmark)
        print(env.ir)  # For debugging in case of error.
        assert instcount["TotalInstsCount"] > 0


def test_random_benchmark(llvm_stress_dataset: LlvmStressDataset):
    num_benchmarks = 5
    rng = np.random.default_rng(0)
    random_benchmarks = {
        b.uri
        for b in (
            llvm_stress_dataset.random_benchmark(rng) for _ in range(num_benchmarks)
        )
    }
    assert len(random_benchmarks) == num_benchmarks


if __name__ == "__main__":
    main()
