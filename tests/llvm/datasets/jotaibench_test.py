# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the JotaiBench dataset."""
import sys
from itertools import islice
from pathlib import Path

import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import JotaiBenchDataset
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def jotaibench_dataset() -> JotaiBenchDataset:
    with gym.make("llvm-v0") as env:
        ds = env.datasets["jotaibench-v0"]
    yield ds


def test_jotaibench_size(jotaibench_dataset: JotaiBenchDataset):
    if sys.platform == "darwin":
        assert jotaibench_dataset.size == 2138894
    else:
        assert jotaibench_dataset.size == 2138894


def test_missing_benchmark_name(jotaibench_dataset: JotaiBenchDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(jotaibench_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://jotaibench-v0$"
    ):
        jotaibench_dataset.benchmark("benchmark://jotaibench-v0")
    jotaibench_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://jotaibench-v0/$"
    ):
        jotaibench_dataset.benchmark("benchmark://jotaibench-v0/")
    assert jotaibench_dataset.install.call_count == 2


@skip_on_ci
@pytest.mark.parametrize("index", range(250))
def test_anghabench_random_select(
    env: LlvmEnv, jotaibench_dataset: JotaiBenchDataset, index: int, tmpwd: Path
):
    uri = next(islice(jotaibench_dataset.benchmark_uris(), index, None))
    benchmark = jotaibench_dataset.benchmark(uri)
    env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "function.c").is_file()


if __name__ == "__main__":
    main()
