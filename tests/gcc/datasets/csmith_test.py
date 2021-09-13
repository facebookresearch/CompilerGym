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

from compiler_gym.envs.gcc.datasets import CsmithBenchmark, CsmithDataset
from tests.pytest_plugins.common import is_ci
from tests.pytest_plugins.gcc import with_gcc_support
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.gcc"]


@pytest.fixture(scope="module")
def csmith_dataset() -> CsmithDataset:
    with gym.make("gcc-v0") as env:
        ds = env.datasets["generator://csmith-v0"]
    yield ds


@with_gcc_support
def test_csmith_size(csmith_dataset: CsmithDataset):
    assert csmith_dataset.size == 0
    assert len(csmith_dataset) == 0


@with_gcc_support
@pytest.mark.parametrize("index", range(3) if is_ci() else range(10))
def test_csmith_random_select(csmith_dataset: CsmithDataset, index: int, tmpwd: Path):
    uri = next(islice(csmith_dataset.benchmark_uris(), index, None))
    benchmark = csmith_dataset.benchmark(uri)
    assert isinstance(benchmark, CsmithBenchmark)
    with gym.make("gcc-v0") as env:
        env.reset(benchmark=benchmark)

    assert benchmark.source
    benchmark.write_sources_to_directory(tmpwd)
    assert (tmpwd / "source.c").is_file()


@with_gcc_support
def test_random_benchmark(csmith_dataset: CsmithDataset):
    num_benchmarks = 5
    rng = np.random.default_rng(0)
    random_benchmarks = {
        b.uri
        for b in (csmith_dataset.random_benchmark(rng) for _ in range(num_benchmarks))
    }
    assert len(random_benchmarks) == num_benchmarks


@with_gcc_support
def test_csmith_from_seed_retry_count_exceeded(csmith_dataset: CsmithDataset):
    with pytest.raises(OSError, match="Csmith failed after 5 attempts with seed 1"):
        csmith_dataset.benchmark_from_seed(seed=1, max_retries=3, retry_count=5)


if __name__ == "__main__":
    main()
