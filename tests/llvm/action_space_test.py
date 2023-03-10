# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the LLVM environment action space."""
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_to_and_from_string_no_actions(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    assert env.action_space.to_string(env.actions) == "opt  input.bc -o output.bc"
    assert env.action_space.from_string(env.action_space.to_string(env.actions)) == []


def test_to_and_from_string(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    env.step(env.action_space.flags.index("-mem2reg"))
    env.step(env.action_space.flags.index("-reg2mem"))
    assert (
        env.action_space.to_string(env.actions)
        == "opt -mem2reg -reg2mem input.bc -o output.bc"
    )
    assert env.action_space.from_string(env.action_space.to_string(env.actions)) == [
        env.action_space.flags.index("-mem2reg"),
        env.action_space.flags.index("-reg2mem"),
    ]


if __name__ == "__main__":
    main()
