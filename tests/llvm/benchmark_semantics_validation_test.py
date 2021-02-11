# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm.datasets import get_llvm_benchmark_validation_callback
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_no_validation_callback_for_custom_benchmark(env: LlvmEnv):
    """Test that a custom benchmark has no validation callback."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "example.c"
        with open(p, "w") as f:
            print("int main() {return 0;}", file=f)
        benchmark = env.make_benchmark(p)

    env.benchmark = benchmark
    env.reset()

    validation_cb = get_llvm_benchmark_validation_callback(env)

    assert validation_cb is None


def test_validate_unoptimized_benchmark(env: LlvmEnv, validatable_benchmark_name: str):
    """Run the validation routine on unoptimized versions of all benchmarks."""
    env.reset(benchmark=validatable_benchmark_name)
    validation_cb = get_llvm_benchmark_validation_callback(env)

    assert validation_cb
    assert validation_cb(env) is None


def test_non_validatable_benchmark_callback(
    env: LlvmEnv, non_validatable_benchmark_name: str
):
    """Run the validation routine on unoptimized versions of all benchmarks."""
    env.reset(benchmark=non_validatable_benchmark_name)
    validation_cb = get_llvm_benchmark_validation_callback(env)

    assert validation_cb is None


if __name__ == "__main__":
    main()
