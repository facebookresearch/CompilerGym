# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM benchmark handling."""
import pytest

from compiler_gym.envs import llvm
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.errors import BenchmarkInitError
from compiler_gym.util.runfiles_path import runfiles_path
from tests.pytest_plugins.common import bazel_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

INVALID_IR_PATH = runfiles_path("tests/llvm/invalid_ir.ll")


@bazel_only  # invalid_ir.ll not installed
def test_reset_invalid_ir(env: LlvmEnv):
    """Test that setting the $CXX to an invalid binary raises an error."""
    benchmark = llvm.make_benchmark(INVALID_IR_PATH)

    with pytest.raises(BenchmarkInitError, match="Failed to compute .text size cost"):
        env.reset(benchmark=benchmark)


if __name__ == "__main__":
    main()
