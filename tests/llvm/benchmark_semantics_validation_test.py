# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

from compiler_gym import ValidationResult
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


def test_validate_benchmark_semantics(env: LlvmEnv, validatable_benchmark_name: str):
    """Run the validation routine on all benchmarks."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark=validatable_benchmark_name)

    # Run a single step.
    env.step(env.action_space.flags.index("-mem2reg"))

    # Validate the environment state.
    result: ValidationResult = env.validate()
    assert not result.error_details
    assert result.reward_validated
    assert not result.actions_replay_failed
    assert not result.reward_validation_failed
    assert result.benchmark_semantics_validated
    assert not result.benchmark_semantics_validation_failed
    assert result.okay()


def test_non_validatable_benchmark_validate(
    env: LlvmEnv, non_validatable_benchmark_name: str
):
    """Run the validation routine on all benchmarks."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark=non_validatable_benchmark_name)

    # Run a single step.
    env.step(env.action_space.flags.index("-mem2reg"))

    # Validate the environment state.
    result: ValidationResult = env.validate()
    assert not result.error_details
    assert result.reward_validated
    assert not result.actions_replay_failed
    assert not result.reward_validation_failed
    assert not result.benchmark_semantics_validated
    assert not result.benchmark_semantics_validation_failed
    assert result.okay()


if __name__ == "__main__":
    main()
