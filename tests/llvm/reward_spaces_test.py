# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import numpy as np
import pytest

from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# Instruction counts for cbench-v1/crc32 benchmark that are used for testing
# reward signals.
CRC32_INSTRUCTION_COUNT = 242
CRC32_INSTRUCTION_COUNT_AFTER_REG2MEM = 249
CRC32_INSTRUCTION_COUNT_O3 = 164
CRC32_INSTRUCTION_COUNT_OZ = 114


def test_instruction_count_reward(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    assert env.observation.IrInstructionCount() == CRC32_INSTRUCTION_COUNT
    action = env.action_space.flags.index("-reg2mem")
    env.step(action)
    assert env.observation.IrInstructionCount() == CRC32_INSTRUCTION_COUNT_AFTER_REG2MEM

    ic_diff = CRC32_INSTRUCTION_COUNT - CRC32_INSTRUCTION_COUNT_AFTER_REG2MEM
    assert env.reward.IrInstructionCount() == ic_diff
    assert env.reward.IrInstructionCountNorm() == ic_diff / CRC32_INSTRUCTION_COUNT

    o3_improvement = CRC32_INSTRUCTION_COUNT - CRC32_INSTRUCTION_COUNT_O3
    assert env.reward.IrInstructionCountO3() == ic_diff / o3_improvement

    oz_improvement = CRC32_INSTRUCTION_COUNT - CRC32_INSTRUCTION_COUNT_OZ
    assert env.reward.IrInstructionCountOz() == ic_diff / oz_improvement


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
    env.reset(benchmark="cbench-v1/crc32")
    invalid = "invalid value"
    with pytest.raises(KeyError) as ctx:
        _ = env.reward[invalid]
    assert str(ctx.value) == f"'{invalid}'"


def test_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    assert set(env.reward.spaces.keys()) == {
        "IrInstructionCount",
        "IrInstructionCountNorm",
        "IrInstructionCountO3",
        "IrInstructionCountOz",
        "ObjectTextSizeBytes",
        "ObjectTextSizeNorm",
        "ObjectTextSizeO3",
        "ObjectTextSizeOz",
    }


def test_instruction_count_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    key = "IrInstructionCount"
    space = env.reward.spaces[key]
    assert str(space) == "IrInstructionCount"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold is None
    assert space.reward_on_error(episode_reward=5) == -5

    key = "IrInstructionCountNorm"
    space = env.reward.spaces[key]
    assert str(space) == "IrInstructionCountNorm"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, 1.0)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold is None
    assert space.reward_on_error(episode_reward=5) == -5

    key = "IrInstructionCountO3"
    space = env.reward.spaces[key]
    assert str(space) == "IrInstructionCountO3"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold == 1
    assert space.reward_on_error(episode_reward=5) == -5

    key = "IrInstructionCountOz"
    space = env.reward.spaces[key]
    assert str(space) == "IrInstructionCountOz"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert not space.platform_dependent
    assert space.success_threshold == 1
    assert space.reward_on_error(episode_reward=5) == -5


def test_native_test_size_reward_spaces(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    key = "ObjectTextSizeBytes"
    space = env.reward.spaces[key]
    assert str(space) == "ObjectTextSizeBytes"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold is None
    assert space.reward_on_error(episode_reward=5) == -5

    key = "ObjectTextSizeNorm"
    space = env.reward.spaces[key]
    assert str(space) == "ObjectTextSizeNorm"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, 1.0)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold is None
    assert space.reward_on_error(episode_reward=5) == -5

    key = "ObjectTextSizeO3"
    space = env.reward.spaces[key]
    assert str(space) == "ObjectTextSizeO3"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold == 1
    assert space.reward_on_error(episode_reward=5) == -5

    key = "ObjectTextSizeOz"
    space = env.reward.spaces[key]
    assert str(space) == "ObjectTextSizeOz"
    assert env.reward[key] == 0
    assert space.range == (-np.inf, np.inf)
    assert space.deterministic
    assert space.platform_dependent
    assert space.success_threshold == 1
    assert space.reward_on_error(episode_reward=5) == -5


if __name__ == "__main__":
    main()
