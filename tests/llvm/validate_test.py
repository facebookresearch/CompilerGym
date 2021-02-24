# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM environment validation."""
import gym

import compiler_gym  # noqa Register environments
from compiler_gym import CompilerEnvState
from tests.test_main import main


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


if __name__ == "__main__":
    main()
