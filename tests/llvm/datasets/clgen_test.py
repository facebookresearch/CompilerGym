# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the CLgen dataset."""
from itertools import islice
from pathlib import Path

import gym
import pytest

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import CLgenDataset
from tests.pytest_plugins.common import is_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def clgen_dataset() -> CLgenDataset:
    env = gym.make("llvm-v0")
    try:
        ds = env.datasets["benchmark://clgen-v0"]
    finally:
        env.close()
    yield ds


def test_clgen_size(clgen_dataset: CLgenDataset):
    assert clgen_dataset.size == 996


def test_missing_benchmark_name(clgen_dataset: CLgenDataset, mocker):
    # Mock install() so that on CI it doesn't download and unpack the tarfile.
    mocker.patch.object(clgen_dataset, "install")

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://clgen-v0$"
    ):
        clgen_dataset.benchmark("benchmark://clgen-v0")
    clgen_dataset.install.assert_called_once()

    with pytest.raises(
        LookupError, match=r"^No benchmark specified: benchmark://clgen-v0/$"
    ):
        clgen_dataset.benchmark("benchmark://clgen-v0/")
    assert clgen_dataset.install.call_count == 2


@pytest.mark.parametrize("index", range(3) if is_ci() else range(250))
def test_clgen_random_select(
    env: LlvmEnv, clgen_dataset: CLgenDataset, index: int, tmpwd: Path
):
    uri = next(islice(clgen_dataset.benchmark_uris(), index, None))
    benchmark = clgen_dataset.benchmark(uri)
    env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "kernel.cl").is_file()


if __name__ == "__main__":
    main()
