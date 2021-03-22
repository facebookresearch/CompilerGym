# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/envs/llvm:legacy_datasets."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv, legacy_datasets
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


def test_validate_sha_output_okay():
    output = legacy_datasets.BenchmarkExecutionResult(
        walltime_seconds=0,
        output="1234567890abcdef 1234567890abcd 1234567890abc 1234567890 12345",
    )
    assert legacy_datasets.validate_sha_output(output) is None


def test_validate_sha_output_invalid():
    output = legacy_datasets.BenchmarkExecutionResult(walltime_seconds=0, output="abcd")
    assert legacy_datasets.validate_sha_output(output)


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
