# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the AnghaBench dataset."""
import sys
from itertools import islice
from pathlib import Path

import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import AnghaBenchDataset
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def anghabench_dataset() -> AnghaBenchDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["anghabench-v1"]
    finally:
        env.close()
    yield ds


def test_anghabench_size(anghabench_dataset: AnghaBenchDataset):
    if sys.platform == "darwin":
        assert anghabench_dataset.size == 1041265
    else:
        assert anghabench_dataset.size == 1041333


def test_missing_benchmark_name(anghabench_dataset: AnghaBenchDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(anghabench_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://anghabench-v1$"
    ):
        anghabench_dataset.benchmark("benchmark://anghabench-v1")
    anghabench_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://anghabench-v1/$"
    ):
        anghabench_dataset.benchmark("benchmark://anghabench-v1/")
    assert anghabench_dataset.install.call_count == 2


@skip_on_ci
@pytest.mark.parametrize("index", range(250))
def test_anghabench_random_select(
    env: LlvmEnv, anghabench_dataset: AnghaBenchDataset, index: int, tmpwd: Path
):
    uri = next(islice(anghabench_dataset.benchmark_uris(), index, None))
    benchmark = anghabench_dataset.benchmark(uri)
    env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "function.c").is_file()


if __name__ == "__main__":
    main()
