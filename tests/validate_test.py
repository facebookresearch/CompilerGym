# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym:validate."""
import gym

from compiler_gym import validate_state, validate_states
from compiler_gym.envs import CompilerEnvState
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
        result = validate_state(env, state)
    finally:
        env.close()

    assert result.success
    assert not result.failed
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
        result = validate_state(env, state)
    finally:
        env.close()

    assert result.success
    assert not result.failed
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
        result = validate_state(env, state)
    finally:
        env.close()

    assert not result.success
    assert result.failed
    assert result.reward_validated
    assert result.reward_validation_failed
    assert (
        str(result)
        == "❌  cBench-v0/crc32  Expected reward 1.0000 but received reward 0.0000"
    )


def test_validate_states_lambda_callback():
    state = CompilerEnvState(
        benchmark="cBench-v0/crc32",
        walltime=1,
        commandline="opt  input.bc -o output.bc",
    )
    results = list(
        validate_states(
            make_env=lambda: gym.make("llvm-v0"), states=[state], datasets=["cBench-v0"]
        )
    )
    assert len(results) == 1
    assert results[0].success


if __name__ == "__main__":
    main()
