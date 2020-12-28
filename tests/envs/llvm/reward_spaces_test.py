# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import numpy as np
import pytest

from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


def test_reward_space(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    assert env.reward_space.id == "IrInstructionCount"

    env.reward_space = None
    assert env.reward_space is None

    invalid = "invalid value"
    with pytest.raises(LookupError) as ctx:
        env.reward_space = invalid
    assert str(ctx.value) == f"Reward space not found: {invalid}"


def test_invalid_reward_space_name(env: LlvmEnv):
    env.reset(benchmark="cBench-v0/crc32")
    invalid = "invalid value"
    with pytest.raises(KeyError) as ctx:
        _ = env.reward[invalid]
    assert str(ctx.value) == f"'{invalid}'"


def test_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cBench-v0/crc32")

    assert set(env.reward.spaces.keys()) == {
        "IrInstructionCount",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
        "ObjectTextSizeBytes",
        "ObjectTextSizeO3",
        "ObjectTextSizeOz",
    }


def test_instruction_count_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cBench-v0/crc32")

    key = "IrInstructionCount"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(IrInstructionCount)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold is None

    key = "IrInstructionCountO3"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(IrInstructionCountO3)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold == 1

    key = "IrInstructionCountOz"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(IrInstructionCountOz)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold == 1


def test_native_test_size_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cBench-v0/crc32")

    key = "ObjectTextSizeBytes"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(ObjectTextSizeBytes)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold is None

    key = "ObjectTextSizeO3"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(ObjectTextSizeO3)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold == 1

    key = "ObjectTextSizeOz"
    space = env.reward.spaces[key]
    assert str(space) == "RewardSpaceSpec(ObjectTextSizeOz)"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold == 1


if __name__ == "__main__":
    main()
