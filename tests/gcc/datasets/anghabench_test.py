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

import compiler_gym.envs.gcc  # noqa register environments
from compiler_gym.envs.gcc import GccEnv
from compiler_gym.envs.gcc.datasets import AnghaBenchDataset
from tests.pytest_plugins.common import skip_on_ci, with_docker
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.gcc"]


@pytest.fixture(scope="module")
def anghabench_dataset() -> AnghaBenchDataset:
    env = gym.make("gcc-v0")
    try:
        ds = env.datasets["anghabench-v1"]
    finally:
        env.close()
    yield ds


@with_docker
def test_anghabench_size(anghabench_dataset: AnghaBenchDataset):
    if sys.platform == "darwin":
        assert anghabench_dataset.size == 1041265
    else:
        assert anghabench_dataset.size == 1041333


@with_docker
def test_missing_benchmark_name(anghabench_dataset: AnghaBenchDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(anghabench_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^Benchmark not found: benchmark://anghabench-v1"
    ):
        anghabench_dataset.benchmark("benchmark://anghabench-v1")
    anghabench_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^Benchmark not found: benchmark://anghabench-v1/"
    ):
        anghabench_dataset.benchmark("benchmark://anghabench-v1/")
    assert anghabench_dataset.install.call_count == 2


@with_docker
@skip_on_ci
@pytest.mark.parametrize("index", range(10))
def test_anghabench_random_select(
    env: GccEnv, anghabench_dataset: AnghaBenchDataset, index: int, tmpwd: Path
):
    uri = next(islice(anghabench_dataset.benchmark_uris(), index, None))
    benchmark = anghabench_dataset.benchmark(uri)
    env.reset(benchmark=benchmark)


if __name__ == "__main__":
    main()
