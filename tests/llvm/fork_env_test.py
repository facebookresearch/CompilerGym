# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LlvmEnv.fork()."""
import pytest

from compiler_gym.envs import LlvmEnv
from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

EXAMPLE_BITCODE_FILE = runfiles_path(
    "compiler_gym/third_party/cBench/cBench-v0/crc32.bc"
)
EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT = 196


def test_fork_state(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    env.step(0)
    assert env.actions == [0]

    new_env = env.fork()
    try:
        assert new_env.benchmark == new_env.benchmark
        assert new_env.actions == env.actions
    finally:
        new_env.close()


def test_fork_reset(env: LlvmEnv):
    env.reset("cBench-v0/crc32")
    env.step(0)
    env.step(1)
    env.step(2)

    new_env = env.fork()
    try:
        new_env.step(3)

        assert env.actions == [0, 1, 2]
        assert new_env.actions == [0, 1, 2, 3]

        new_env.reset()
        assert env.actions == [0, 1, 2]
        assert new_env.actions == []
    finally:
        new_env.close()


def test_fork_custom_benchmark(env: LlvmEnv):
    benchmark = env.make_benchmark(EXAMPLE_BITCODE_FILE)
    env.reset(benchmark=benchmark)

    def ir(env):
        """Strip the ModuleID line from IR."""
        return "\n".join(env.ir.split("\n")[1:])

    try:
        new_env = env.fork()
        assert ir(env) == ir(new_env)

        new_env.reset()
        assert ir(env) == ir(new_env)
    finally:
        new_env.close()


def test_fork_twice_test(env: LlvmEnv):
    """Test that fork() on a forked environment works."""
    env.reset(benchmark="cBench-v0/crc32")
    fork_a = env.fork()
    try:
        fork_b = fork_a.fork()
        try:
            assert env.state == fork_a.state
            assert fork_a.state == fork_b.state
        finally:
            fork_b.close()
    finally:
        fork_a.close()


@pytest.mark.xfail(reason="Scaled rewards use the incorrect scale for fork()")
def test_fork_rewards(env: LlvmEnv, reward_space: str):
    """Test that rewards are equal after fork() is called."""
    env.reward_space = reward_space
    env.reset("cBench-v0/dijkstra")

    actions = env.action_space.names
    act_indcs = [actions.index(n) for n in ["-mem2reg", "-simplifycfg"]]

    for i in range(len(act_indcs)):
        act_indc = act_indcs[i]

        forked = env.fork()
        try:
            _, env_reward, _, _ = env.step(act_indc)
            _, fkd_reward, _, _ = forked.step(act_indc)
            assert env_reward == fkd_reward
        finally:
            forked.close()


if __name__ == "__main__":
    main()
