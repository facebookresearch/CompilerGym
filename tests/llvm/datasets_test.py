# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/envs/llvm:datasets."""
import os

import gym
import pytest

from compiler_gym.envs.llvm import LlvmEnv, datasets
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


def test_validate_sha_output_okay():
    output = datasets.BenchmarkExecutionResult(
        walltime_seconds=0,
        output="1234567890abcdef 1234567890abcd 1234567890abc 1234567890 12345",
    )
    assert datasets.validate_sha_output(output) is None


def test_validate_sha_output_invalid():
    output = datasets.BenchmarkExecutionResult(walltime_seconds=0, output="abcd")
    assert datasets.validate_sha_output(output)


def test_default_cBench_dataset_require(tmpwd, temporary_environ):
    """Test that cBench is downloaded."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")
    try:
        assert not env.benchmarks, "Sanity check"

        # Datasaet is downloaded.
        assert env.require_dataset("cBench-v1")
        assert env.benchmarks

        # Dataset is already downloaded.
        assert not env.require_dataset("cBench-v1")
    finally:
        env.close()


def test_default_cBench_on_reset(tmpwd, temporary_environ):
    """Test that cBench is downloaded by default when no benchmarks are available."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")
    try:
        assert not env.benchmarks, "Sanity check"

        env.reset()
        assert env.benchmarks
        assert env.benchmark.startswith("benchmark://cBench-v1/")
    finally:
        env.close()


@pytest.mark.parametrize("benchmark_name", ["benchmark://npb-v0/1", "npb-v0/1"])
def test_dataset_required(tmpwd, temporary_environ, benchmark_name):
    """Test that the required dataset is downlaoded when a benchmark is specified."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")
    try:
        env.reset(benchmark=benchmark_name)

        assert env.benchmarks
        assert env.benchmark.startswith("benchmark://npb-v0/")
    finally:
        env.close()


def test_cBench_v0_deprecation(env: LlvmEnv):
    """Test that cBench-v0 emits a deprecation warning when used."""
    with pytest.deprecated_call(
        match=(
            "Dataset 'cBench-v0' is deprecated as of CompilerGym release "
            "v0.1.4, please update to the latest available version"
        )
    ):
        env.require_dataset("cBench-v0")


if __name__ == "__main__":
    main()
