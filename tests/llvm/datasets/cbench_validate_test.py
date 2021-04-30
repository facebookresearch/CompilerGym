# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test for cBench semantics validation."""
from compiler_gym import ValidationResult
from compiler_gym.envs.llvm import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_validate_benchmark_semantics(env: LlvmEnv, validatable_cbench_uri: str):
    """Run the validation routine on all benchmarks."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark=validatable_cbench_uri)

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
    env: LlvmEnv, non_validatable_cbench_uri: str
):
    """Run the validation routine on all benchmarks."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark=non_validatable_cbench_uri)

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
