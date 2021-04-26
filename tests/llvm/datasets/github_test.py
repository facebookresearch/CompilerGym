# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GitHub dataset."""
import sys
from itertools import islice

import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import GitHubDataset
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def github_dataset() -> GitHubDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["github-v0"]
    finally:
        env.close()
    yield ds


def test_github_size(github_dataset: GitHubDataset):
    if sys.platform == "linux":
        assert github_dataset.size == 49738
    else:
        assert github_dataset.size == 47806


@skip_on_ci
@pytest.mark.parametrize("index", range(250))
def test_github_random_select(env: LlvmEnv, github_dataset: GitHubDataset, index: int):
    uri = next(islice(github_dataset.benchmark_uris(), index, None))
    benchmark = github_dataset.benchmark(uri)
    env.reset(benchmark=benchmark)


if __name__ == "__main__":
    main()
