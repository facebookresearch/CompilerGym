# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the JotaiBench dataset."""
import sys

import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments

# from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import JotaiBenchDataset

# from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

# from itertools import islice
# from pathlib import Path


pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def jotaibench_dataset() -> JotaiBenchDataset:
    with gym.make("llvm-v0") as env:
        ds = env.datasets["jotaibench-v1"]
    yield ds


def test_jotaibench_size(jotaibench_dataset: JotaiBenchDataset):
    if sys.platform == "darwin":
        assert jotaibench_dataset.size == 2138885
    else:
        assert jotaibench_dataset.size == 2138885


def test_missing_benchmark_name(jotaibench_dataset: JotaiBenchDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(jotaibench_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://jotaibench-v1$"
    ):
        jotaibench_dataset.benchmark("benchmark://jotaibench-v1")
    jotaibench_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://jotaibench-v1/$"
    ):
        jotaibench_dataset.benchmark("benchmark://jotaibench-v1/")
    assert jotaibench_dataset.install.call_count == 2


if __name__ == "__main__":
    main()
