# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the matmul dataset."""
import re
from copy import deepcopy
from itertools import islice
from pathlib import Path

import gym
import numpy as np
import pytest

import compiler_gym.envs.mlir  # noqa register environments
from compiler_gym.envs.mlir import MlirEnv
from compiler_gym.envs.mlir.datasets import MatmulBenchmark, MatmulDataset
from tests.pytest_plugins.common import is_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.mlir"]


@pytest.fixture(scope="module")
def matmul_dataset() -> MatmulDataset:
    with gym.make("mlir-v0") as env:
        ds = env.datasets["generator://matmul-v0"]
    yield ds


def test_matmul_size(matmul_dataset: MatmulDataset):
    assert matmul_dataset.size == 1
    assert len(matmul_dataset) == 1


@pytest.mark.parametrize("index", range(1) if is_ci() else range(1))
def test_matmul_random_select(
    env: MlirEnv, matmul_dataset: MatmulDataset, index: int, tmpwd: Path
):
    uri = next(islice(matmul_dataset.benchmark_uris(), index, None))
    benchmark = matmul_dataset.benchmark(uri)
    assert isinstance(benchmark, MatmulBenchmark)
    env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "source.mlir").is_file()


def test_matmul_from_seed_retry_count_exceeded(matmul_dataset: MatmulDataset):
    with pytest.raises(
        OSError, match=re.escape("matmul failed after 5 attempts with size (4, 4, 4)")
    ):
        matmul_dataset.benchmark_from_size(mnk=(4, 4, 4), max_retries=3, retry_count=5)


def test_matmul_positive_runtimes(env: MlirEnv, matmul_dataset: MatmulDataset):
    benchmark = next(matmul_dataset.benchmarks())
    env.reset(benchmark=benchmark)
    action_space = deepcopy(env.action_space)
    action_space.seed(123)
    env.step(action_space.sample())
    val = env.observation["Runtime"]
    assert np.all(np.greater(val, 0))


if __name__ == "__main__":
    main()
