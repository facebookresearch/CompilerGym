# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

import gym
import pytest

from compiler_gym import CompilerEnvState
from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_validate_state_no_reward():
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v1/crc32",
        walltime=1,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0")
    try:
        result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert not result.reward_validated
    assert str(result) == "✅  cbench-v1/crc32"


def test_validate_state_with_reward():
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v1/crc32",
        walltime=1,
        reward=0,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert result.reward_validated
    assert not result.reward_validation_failed
    assert str(result) == "✅  cbench-v1/crc32  0.0000"


def test_validate_state_invalid_reward():
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v1/crc32",
        walltime=1,
        reward=1,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        result = env.validate(state)
    finally:
        env.close()

    assert not result.okay()
    assert result.reward_validated
    assert result.reward_validation_failed
    assert (
        str(result) == "❌  cbench-v1/crc32  Expected reward 1.0 but received reward 0.0"
    )


def test_validate_state_without_state_reward():
    """Validating state when state has no reward value."""
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v1/crc32",
        walltime=1,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert not result.reward_validated
    assert not result.reward_validation_failed


def test_validate_state_without_env_reward():
    """Validating state when environment has no reward space."""
    state = CompilerEnvState(
        benchmark="benchmark://cbench-v1/crc32",
        walltime=1,
        reward=0,
        commandline="opt  input.bc -o output.bc",
    )
    env = gym.make("llvm-v0")
    try:
        with pytest.warns(
            UserWarning,
            match=(
                "Validating state with reward, "
                "but environment has no reward space set"
            ),
        ):
            result = env.validate(state)
    finally:
        env.close()

    assert result.okay()
    assert not result.reward_validated
    assert not result.reward_validation_failed


def test_no_validation_callback_for_custom_benchmark(env: LlvmEnv):
    """Test that a custom benchmark has no validation callback."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "example.c"
        with open(p, "w") as f:
            print("int main() {return 0;}", file=f)
        benchmark = env.make_benchmark(p)

    env.reset(benchmark=benchmark)

    assert not env.benchmark.is_validatable()


if __name__ == "__main__":
    main()
