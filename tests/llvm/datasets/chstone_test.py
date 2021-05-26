# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the AnghaBench dataset."""
import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import CHStoneDataset, chstone
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def chstone_dataset() -> CHStoneDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["chstone-v0"]
    finally:
        env.close()
    yield ds


def test_anghabench_size(chstone_dataset: CHStoneDataset):
    assert chstone_dataset.size == 12


def test_missing_benchmark_name(chstone_dataset: CHStoneDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(chstone_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://chstone-v0$"
    ):
        chstone_dataset.benchmark("benchmark://chstone-v0")
    chstone_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://chstone-v0/$"
    ):
        chstone_dataset.benchmark("benchmark://chstone-v0/")
    assert chstone_dataset.install.call_count == 2


@pytest.mark.parametrize("uri", chstone.URIS)
def test_chstone_benchmark_reset(
    env: LlvmEnv, chstone_dataset: CHStoneDataset, uri: str
):
    env.reset(chstone_dataset.benchmark(uri))
    assert env.benchmark == uri


if __name__ == "__main__":
    main()
