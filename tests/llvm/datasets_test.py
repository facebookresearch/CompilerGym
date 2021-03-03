# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/envs/llvm:datasets."""

from compiler_gym.envs.llvm import datasets
from tests.test_main import main


def test_validate_sha_output_okay():
    output = datasets.BenchmarkExecutionResult(
        walltime_seconds=0,
        output="1234567890abcdef 1234567890abcd 1234567890abc 1234567890 12345".encode(
            "utf-8"
        ),
    )
    assert datasets.validate_sha_output(output) is None


def test_validate_sha_output_invalid():
    output = datasets.BenchmarkExecutionResult(
        walltime_seconds=0, output="abcd".encode("utf-8")
    )
    assert datasets.validate_sha_output(output)


if __name__ == "__main__":
    main()
