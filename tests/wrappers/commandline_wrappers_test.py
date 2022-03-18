# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import CommandlineWithTerminalAction, ConstrainedCommandline
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_commandline_with_terminal_action(env: LlvmEnv):
    mem2reg_unwrapped_index = env.action_space["-mem2reg"]

    env = CommandlineWithTerminalAction(env)

    mem2reg_index = env.action_space["-mem2reg"]
    reg2mem_index = env.action_space["-reg2mem"]

    assert mem2reg_index == mem2reg_unwrapped_index

    env.reset()
    _, _, done, info = env.step(mem2reg_index)
    assert not done, info
    _, _, done, info = env.multistep([reg2mem_index, reg2mem_index])
    assert not done, info

    assert env.actions == [mem2reg_index, reg2mem_index, reg2mem_index]

    _, _, done, info = env.step(len(env.action_space.flags) - 1)
    assert done
    assert "terminal_action" in info


def test_commandline_with_terminal_action_fork(env: LlvmEnv):
    env = CommandlineWithTerminalAction(env)
    assert env.unwrapped.action_space != env.action_space  # Sanity check.
    with env.fork() as fkd:
        assert fkd.action_space == env.action_space

        _, _, done, _ = env.step(len(env.action_space.flags) - 1)
        assert done

        _, _, done, _ = fkd.step(len(env.action_space.flags) - 1)
        assert done


def test_constrained_action_space(env: LlvmEnv):
    mem2reg_index = env.action_space["-mem2reg"]
    reg2mem_index = env.action_space["-reg2mem"]

    env = ConstrainedCommandline(env=env, flags=["-mem2reg", "-reg2mem"])

    assert env.action_space.n == 2
    assert env.action_space.flags == ["-mem2reg", "-reg2mem"]

    assert env.action(0) == mem2reg_index
    assert env.action([0, 1]) == [mem2reg_index, reg2mem_index]

    env.reset()
    env.step(0)
    env.multistep([1, 1])

    assert env.actions == [0, 1, 1]


def test_constrained_action_space_fork(env: LlvmEnv):
    mem2reg_index = env.action_space["-mem2reg"]
    reg2mem_index = env.action_space["-reg2mem"]

    env = ConstrainedCommandline(env=env, flags=["-mem2reg", "-reg2mem"])

    fkd = env.fork()
    try:
        assert fkd.action_space.n == 2
        assert fkd.action_space.flags == ["-mem2reg", "-reg2mem"]

        assert fkd.action(0) == mem2reg_index
        assert fkd.action([0, 1]) == [mem2reg_index, reg2mem_index]

        fkd.reset()
        fkd.step(0)
        fkd.multistep([1, 1])

        assert fkd.actions == [0, 1, 1]
    finally:
        fkd.close()


if __name__ == "__main__":
    main()
