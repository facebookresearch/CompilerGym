# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

import gym

from compiler_gym import CompilerEnvState, ValidationResult
from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm.datasets import get_llvm_benchmark_validation_callback
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_validate_state_no_reward():
    state = CompilerEnvState(
        benchmark="cBench-v0/crc32",
        walltime=1,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0")
    try:
        env.require_dataset("cBench-v0")
        result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert not result.reward_validated
    assert str(result) == "✅  cBench-v0/crc32"


def test_validate_state_with_reward():
    state = CompilerEnvState(
        benchmark="cBench-v0/crc32",
        walltime=1,
        reward=0,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        env.require_dataset("cBench-v0")
        result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert result.reward_validated
    assert not result.reward_validation_failed
    assert str(result) == "✅  cBench-v0/crc32  0.0000"


def test_validate_state_invalid_reward():
    state = CompilerEnvState(
        benchmark="cBench-v0/crc32",
        walltime=1,
        reward=1,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        env.require_dataset("cBench-v0")
        result = env.validate(state)
    finally:
        env.close()

    assert not result.okay()
    assert result.reward_validated
    assert result.reward_validation_failed
    assert (
        str(result)
        == "❌  cBench-v0/crc32  Expected reward 1.0000 but received reward 0.0000"
    )


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
