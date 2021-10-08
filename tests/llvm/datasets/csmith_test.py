# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Csmith dataset."""
from itertools import islice
from pathlib import Path

import gym
import numpy as np
import pytest
from flaky import flaky

import compiler_gym.envs.llvm  # noqa register environments
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import CsmithBenchmark, CsmithDataset
from compiler_gym.service import ServiceError
from tests.pytest_plugins.common import is_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def csmith_dataset() -> CsmithDataset:
    with gym.make("llvm-v0") as env:
        ds = env.datasets["generator://csmith-v0"]
    yield ds


def test_csmith_size(csmith_dataset: CsmithDataset):
    assert csmith_dataset.size == 0
    assert len(csmith_dataset) == 0


@pytest.mark.parametrize("index", range(3) if is_ci() else range(250))
def test_csmith_random_select(
    env: LlvmEnv, csmith_dataset: CsmithDataset, index: int, tmpwd: Path
):
    uri = next(islice(csmith_dataset.benchmark_uris(), index, None))
    benchmark = csmith_dataset.benchmark(uri)
    assert isinstance(benchmark, CsmithBenchmark)
    env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "source.c").is_file()


def test_random_benchmark(csmith_dataset: CsmithDataset):
    num_benchmarks = 5
    rng = np.random.default_rng(0)
    random_benchmarks = {
        b.uri
        for b in (csmith_dataset.random_benchmark(rng) for _ in range(num_benchmarks))
    }
    assert len(random_benchmarks) == num_benchmarks


def test_csmith_from_seed_retry_count_exceeded(csmith_dataset: CsmithDataset):
    with pytest.raises(OSError, match="Csmith failed after 5 attempts with seed 1"):
        csmith_dataset.benchmark_from_seed(seed=1, max_retries=3, retry_count=5)


@flaky(rerun_filter=lambda err, *args: issubclass(err[0], ServiceError))
def test_csmith_positive_runtimes(env: LlvmEnv, csmith_dataset: CsmithDataset):
    benchmark = next(csmith_dataset.benchmarks())
    env.reset(benchmark=benchmark)
    val = env.observation["Runtime"]
    print(val.tolist())
    assert np.all(np.greater(val, 0))


@flaky(rerun_filter=lambda err, *args: issubclass(err[0], ServiceError))
def test_csmith_positive_buildtimes(env: LlvmEnv, csmith_dataset: CsmithDataset):
    benchmark = next(csmith_dataset.benchmarks())
    env.reset(benchmark=benchmark)
    val = env.observation["Buildtime"]
    print(val.tolist())
    assert np.all(np.greater(val, 0))


if __name__ == "__main__":
    main()
