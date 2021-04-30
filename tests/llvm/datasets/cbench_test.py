# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the cbench dataset."""
import tempfile
from pathlib import Path

import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import CBenchDataset, cbench
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="module")
def cbench_dataset() -> CBenchDataset:
    with tempfile.TemporaryDirectory() as d:
        yield CBenchDataset(site_data_base=Path(d))


def test_cbench_size(cbench_dataset: CBenchDataset):
    assert cbench_dataset.size == 23


def test_cbench_uris(cbench_dataset: CBenchDataset):
    assert list(cbench_dataset.benchmark_uris()) == [
        "benchmark://cbench-v1/adpcm",
        "benchmark://cbench-v1/bitcount",
        "benchmark://cbench-v1/blowfish",
        "benchmark://cbench-v1/bzip2",
        "benchmark://cbench-v1/crc32",
        "benchmark://cbench-v1/dijkstra",
        "benchmark://cbench-v1/ghostscript",
        "benchmark://cbench-v1/gsm",
        "benchmark://cbench-v1/ispell",
        "benchmark://cbench-v1/jpeg-c",
        "benchmark://cbench-v1/jpeg-d",
        "benchmark://cbench-v1/lame",
        "benchmark://cbench-v1/patricia",
        "benchmark://cbench-v1/qsort",
        "benchmark://cbench-v1/rijndael",
        "benchmark://cbench-v1/sha",
        "benchmark://cbench-v1/stringsearch",
        "benchmark://cbench-v1/stringsearch2",
        "benchmark://cbench-v1/susan",
        "benchmark://cbench-v1/tiff2bw",
        "benchmark://cbench-v1/tiff2rgba",
        "benchmark://cbench-v1/tiffdither",
        "benchmark://cbench-v1/tiffmedian",
    ]


def test_validate_sha_output_okay():
    output = cbench.BenchmarkExecutionResult(
        walltime_seconds=0,
        output="1234567890abcdef 1234567890abcd 1234567890abc 1234567890 12345",
    )
    assert cbench.validate_sha_output(output) is None


def test_validate_sha_output_invalid():
    output = cbench.BenchmarkExecutionResult(walltime_seconds=0, output="abcd")
    assert cbench.validate_sha_output(output)


def test_cbench_v0_deprecation(env: LlvmEnv):
    """Test that cBench-v0 emits a deprecation warning when used."""
    with pytest.deprecated_call(match="Please use 'benchmark://cbench-v1'"):
        env.datasets["cBench-v0"].install()

    with pytest.deprecated_call(match="Please use 'benchmark://cbench-v1'"):
        env.datasets.benchmark("benchmark://cBench-v0/crc32")


def test_cbench_v1_deprecation(env: LlvmEnv):
    """Test that cBench-v1 emits a deprecation warning when used."""
    with pytest.deprecated_call(match="Please use 'benchmark://cbench-v1'"):
        env.datasets["cBench-v1"].install()

    with pytest.deprecated_call(match="Please use 'benchmark://cbench-v1'"):
        env.datasets.benchmark("benchmark://cBench-v1/crc32")


if __name__ == "__main__":
    main()
