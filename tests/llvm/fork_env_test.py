# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LlvmEnv.fork()."""
import subprocess

from compiler_gym.envs import LlvmEnv
from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

EXAMPLE_BITCODE_FILE = runfiles_path(
    "compiler_gym/third_party/cbench/cbench-v1/crc32.bc"
)
EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT = 196


def test_fork_child_process_is_not_orphaned(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    fkd = env.fork()
    try:
        # Check that both environments share the same service.
        assert isinstance(env.service.connection.process, subprocess.Popen)
        assert isinstance(fkd.service.connection.process, subprocess.Popen)

        assert env.service.connection.process.pid == fkd.service.connection.process.pid
        process = env.service.connection.process

        # Sanity check that both services are alive.
        assert not env.service.connection.process.poll()
        assert not fkd.service.connection.process.poll()

        # Close the parent service.
        env.close()

        # Check that the service is still alive.
        assert not env.service
        assert not fkd.service.connection.process.poll()

        # Close the forked service.
        fkd.close()

        # Check that the service has been killed.
        assert process.poll()
    finally:
        fkd.close()


def test_fork_chain_child_processes_are_not_orphaned(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    # Create a chain of forked environments.
    a = env.fork()
    b = a.fork()
    c = b.fork()
    d = c.fork()

    try:
        # Sanity check that they share the same underlying service.
        assert (
            env.service.connection.process
            == a.service.connection.process
            == b.service.connection.process
            == c.service.connection.process
            == d.service.connection.process
        )
        proc = env.service.connection.process
        # Kill the forked environments one by one.
        a.close()
        assert proc.poll() is None
        b.close()
        assert proc.poll() is None
        c.close()
        assert proc.poll() is None
        d.close()
        assert proc.poll() is None
        # Kill the final environment, refcount 0, service is closed.
        env.close()
        assert proc.poll() is not None
    finally:
        a.close()
        b.close()
        c.close()
        d.close()


def test_fork_before_reset(env: LlvmEnv):
    """Test that fork() before reset() starts an episode."""
    assert not env.in_episode
    fkd = env.fork()
    try:
        assert env.in_episode
        assert fkd.in_episode
    finally:
        fkd.close()


def test_fork_closed_service(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    _, _, done, _ = env.step(0)
    assert not done
    assert env.actions == [0]

    env.close()
    assert not env.service

    fkd = env.fork()
    try:
        assert env.actions == [0]
        assert fkd.actions == [0]
    finally:
        fkd.close()


def test_fork_spaces_are_same(env: LlvmEnv):
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cbench-v1/crc32")

    fkd = env.fork()
    try:
        assert fkd.observation_space == env.observation_space
        assert fkd.reward_space == env.reward_space
        assert fkd.benchmark == env.benchmark
    finally:
        fkd.close()


def test_fork_state(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    env.step(0)
    assert env.actions == [0]

    new_env = env.fork()
    try:
        assert new_env.benchmark == new_env.benchmark
        assert new_env.actions == env.actions
    finally:
        new_env.close()


def test_fork_reset(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
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

    new_env = env.fork()
    try:
        assert ir(env) == ir(new_env)

        new_env.reset()
        assert ir(env) == ir(new_env)
    finally:
        new_env.close()


def test_fork_twice_test(env: LlvmEnv):
    """Test that fork() on a forked environment works."""
    env.reset(benchmark="cbench-v1/crc32")
    fork_a = env.fork()
    fork_b = fork_a.fork()
    try:
        assert env.state == fork_a.state
        assert fork_a.state == fork_b.state
    finally:
        fork_a.close()
        fork_b.close()


def test_fork_modified_ir_is_the_same(env: LlvmEnv):
    """Test that the IR of a forked environment is the same."""
    env.reset("cbench-v1/crc32")

    # Apply an action that modifies the benchmark.
    _, _, done, info = env.step(env.action_space.flags.index("-mem2reg"))
    assert not done
    assert not info["action_had_no_effect"]

    forked = env.fork()
    try:
        assert "\n".join(env.ir.split("\n")[1:]) == "\n".join(forked.ir.split("\n")[1:])

        # Apply another action.
        _, _, done, info = env.step(env.action_space.flags.index("-gvn"))
        _, _, done, info = forked.step(forked.action_space.flags.index("-gvn"))
        assert not done
        assert not info["action_had_no_effect"]

        # Check that IRs are still equivalent.
        assert "\n".join(env.ir.split("\n")[1:]) == "\n".join(forked.ir.split("\n")[1:])
    finally:
        forked.close()


def test_fork_rewards(env: LlvmEnv, reward_space: str):
    """Test that rewards are equal after fork() is called."""
    env.reward_space = reward_space
    env.reset("cbench-v1/dijkstra")

    actions = [env.action_space.flags.index(n) for n in ["-mem2reg", "-simplifycfg"]]

    forked = env.fork()
    try:
        for action in actions:
            _, env_reward, env_done, _ = env.step(action)
            _, fkd_reward, fkd_done, _ = forked.step(action)
            assert env_done is False
            assert fkd_done is False
            assert env_reward == fkd_reward
    finally:
        forked.close()


def test_fork_previous_cost_reward_update(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset("cbench-v1/crc32")

    env.step(env.action_space.flags.index("-mem2reg"))
    fkd = env.fork()
    try:
        _, a, _, _ = env.step(env.action_space.flags.index("-mem2reg"))
        _, b, _, _ = fkd.step(env.action_space.flags.index("-mem2reg"))
        assert a == b
    finally:
        fkd.close()


def test_fork_previous_cost_lazy_reward_update(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    env.step(env.action_space.flags.index("-mem2reg"))
    env.reward["IrInstructionCount"]
    fkd = env.fork()
    try:
        env.step(env.action_space.flags.index("-mem2reg"))
        fkd.step(env.action_space.flags.index("-mem2reg"))

        assert env.reward["IrInstructionCount"] == fkd.reward["IrInstructionCount"]
    finally:
        fkd.close()


if __name__ == "__main__":
    main()
