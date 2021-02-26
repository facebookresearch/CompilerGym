# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the AnghaBench dataset."""
import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm.datasets import AnghaBenchDataset
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def anghabench_dataset() -> AnghaBenchDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["anghabench-v0"]
    finally:
        env.close()
    yield ds


def test_anghabench_count(anghabench_dataset: AnghaBenchDataset):
    assert anghabench_dataset.n == 1044021


@skip_on_ci
@pytest.mark.parametrize("seed", range(250))
def test_anghabench_random_select(
    env: LlvmEnv, anghabench_dataset: AnghaBenchDataset, seed: int
):
    anghabench_dataset.seed(seed)
    benchmark = anghabench_dataset.benchmark()
    env.reset(benchmark=benchmark)


if __name__ == "__main__":
    main()
