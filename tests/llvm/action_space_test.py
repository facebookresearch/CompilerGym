# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the LLVM environment action space."""
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_commandline_no_actions(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    assert env.commandline() == "opt  input.bc -o output.bc"
    assert env.commandline_to_actions(env.commandline()) == []


def test_commandline(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    env.step(env.action_space["-mem2reg"])
    env.step(env.action_space["-reg2mem"])
    assert env.commandline() == "opt -mem2reg -reg2mem input.bc -o output.bc"
    assert env.commandline_to_actions(env.commandline()) == [
        env.action_space["-mem2reg"],
        env.action_space["-reg2mem"],
    ]


if __name__ == "__main__":
    main()
