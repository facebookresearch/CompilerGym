# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the AnghaBench dataset."""
import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import LlvmStressDataset
from tests.pytest_plugins.common import skip_on_ci
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


@skip_on_ci
@pytest.mark.parametrize("seed", range(250))
def test_llvm_stress_random_select(
    env: LlvmEnv, llvm_stress_dataset: LlvmStressDataset, seed: int
):
    llvm_stress_dataset.seed(seed)
    benchmark = llvm_stress_dataset.benchmark()
    env.observation_space = "InstCountDict"
    instcount = env.reset(benchmark=benchmark)
    print(env.ir)  # For debugging in case of error.
    assert instcount["TotalInstsCount"] > 0


if __name__ == "__main__":
    main()
