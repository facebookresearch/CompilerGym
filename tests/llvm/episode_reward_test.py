# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LlvmEnv.episode_reward."""
from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_episode_reward_init_zero(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset("cbench-v1/crc32")
    assert env.episode_reward == 0
    _, reward, _, _ = env.step(env.action_space["-mem2reg"])
    assert reward > 0
    assert env.episode_reward == reward
    env.reset()
    assert env.episode_reward == 0


def test_episode_reward_with_non_default_reward_space(env: LlvmEnv):
    """Test that episode_reward is not updated when custom rewards passed to
    step()."""
    env.reward_space = "IrInstructionCountOz"
    env.reset("cbench-v1/crc32")
    assert env.episode_reward == 0
    _, rewards, _, _ = env.step(
        env.action_space["-mem2reg"],
        rewards=["IrInstructionCount"],
    )
    assert rewards[0] > 0
    assert env.episode_reward == 0


if __name__ == "__main__":
    main()
