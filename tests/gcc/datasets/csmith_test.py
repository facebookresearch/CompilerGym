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

from compiler_gym.envs.gcc.datasets import CsmithBenchmark
from tests.pytest_plugins.common import is_ci
from tests.pytest_plugins.gcc import with_gcc_support
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.gcc"]


@pytest.mark.xfail(
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
@with_gcc_support
def test_csmith_size(gcc_bin: str):
    with gym.make("gcc-v0", gcc_bin=gcc_bin) as env:
        csmith_dataset = env.datasets["generator://csmith-v0"]

        assert csmith_dataset.size == 0
        assert len(csmith_dataset) == 0


@pytest.mark.xfail(
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
@with_gcc_support
@pytest.mark.parametrize("index", range(3) if is_ci() else range(10))
def test_csmith_random_select(gcc_bin: str, index: int, tmpwd: Path):
    with gym.make("gcc-v0", gcc_bin=gcc_bin) as env:
        csmith_dataset = env.datasets["generator://csmith-v0"]

        uri = next(islice(csmith_dataset.benchmark_uris(), index, None))
        benchmark = csmith_dataset.benchmark(uri)
        assert isinstance(benchmark, CsmithBenchmark)
        env.reset(benchmark=benchmark)

        assert benchmark.source
        benchmark.write_sources_to_directory(tmpwd)
        assert (tmpwd / "source.c").is_file()


@pytest.mark.xfail(
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
@with_gcc_support
def test_random_benchmark(gcc_bin: str):
    with gym.make("gcc-v0", gcc_bin=gcc_bin) as env:
        csmith_dataset = env.datasets["generator://csmith-v0"]

        num_benchmarks = 5
        rng = np.random.default_rng(0)
        random_benchmarks = {
            b.uri
            for b in (
                csmith_dataset.random_benchmark(rng) for _ in range(num_benchmarks)
            )
        }
        assert len(random_benchmarks) == num_benchmarks


@pytest.mark.xfail(
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
@with_gcc_support
def test_csmith_from_seed_retry_count_exceeded(gcc_bin: str):
    with gym.make("gcc-v0", gcc_bin=gcc_bin) as env:
        csmith_dataset = env.datasets["generator://csmith-v0"]

        with pytest.raises(OSError, match="Csmith failed after 5 attempts with seed 1"):
            csmith_dataset.benchmark_from_seed(seed=1, max_retries=3, retry_count=5)


if __name__ == "__main__":
    main()
