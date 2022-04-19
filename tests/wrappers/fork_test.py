# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import ForkOnStep
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_ForkOnStep_step(env: LlvmEnv):
    with ForkOnStep(env) as env:
        env.reset()
        assert env.stack == []

        env.step(0)
        assert env.actions == [0]
        assert len(env.stack) == 1
        assert env.stack[0].actions == []

        env.step(1)
        assert env.actions == [0, 1]
        assert len(env.stack) == 2
        assert env.stack[1].actions == [0]
        assert env.stack[0].actions == []


def test_ForkOnStep_reset(env: LlvmEnv):
    with ForkOnStep(env) as env:
        env.reset()

        env.step(0)
        assert env.actions == [0]
        assert len(env.stack) == 1

        env.reset()
        assert env.actions == []
        assert env.stack == []


def test_ForkOnStep_double_close(env: LlvmEnv):
    with ForkOnStep(env) as env:
        env.close()
        env.close()


def test_ForkOnStep_undo(env: LlvmEnv):
    with ForkOnStep(env) as env:
        env.reset()

        env.step(0)
        assert env.actions == [0]
        assert len(env.stack) == 1

        env.undo()
        assert env.actions == []
        assert not env.stack

        # Undo of an empty stack:
        env.undo()
        assert env.actions == []
        assert not env.stack


if __name__ == "__main__":
    main()
