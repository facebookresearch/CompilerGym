# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LlvmEnv.fork()."""
import subprocess
import sys

import pytest

import compiler_gym
from compiler_gym.envs.llvm import LLVM_SERVICE_BINARY, LlvmEnv
from compiler_gym.service import (
    CompilerGymServiceConnection,
    ConnectionOpts,
    ServiceError,
)
from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

EXAMPLE_BITCODE_FILE = runfiles_path(
    "compiler_gym/third_party/cbench/cbench-v1/crc32.bc"
)
EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT = 196


def test_with_statement(env: LlvmEnv):
    """Test that the `with` statement context manager works on forks."""
    env.reset("cbench-v1/crc32")
    env.step(0)
    with env.fork() as fkd:
        assert fkd.in_episode
        assert fkd.actions == [0]
    assert not fkd.in_episode
    assert env.in_episode


def test_fork_child_process_is_not_orphaned():
    service = CompilerGymServiceConnection(LLVM_SERVICE_BINARY, ConnectionOpts())

    with compiler_gym.make("llvm-v0", service_connection=service) as env:
        env.reset("cbench-v1/crc32")
        with env.fork() as fkd:
            # Check that both environments share the same service.
            assert isinstance(env.service.connection.process, subprocess.Popen)
            assert isinstance(fkd.service.connection.process, subprocess.Popen)

            assert (
                env.service.connection.process.pid == fkd.service.connection.process.pid
            )
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
            assert process.poll() is not None


def test_fork_chain_child_processes_are_not_orphaned(env: LlvmEnv):
    service = CompilerGymServiceConnection(LLVM_SERVICE_BINARY, ConnectionOpts())

    with compiler_gym.make("llvm-v0", service_connection=service) as env:
        env.reset()

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
    with env.fork() as fkd:
        assert env.in_episode
        assert fkd.in_episode


def test_fork_closed_service(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")

    _, _, done, _ = env.step(0)
    assert not done
    assert env.actions == [0]

    env.close()
    assert not env.service

    with env.fork() as fkd:
        assert env.actions == [0]
        assert fkd.actions == [0]


def test_fork_spaces_are_same(env: LlvmEnv):
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cbench-v1/crc32")

    with env.fork() as fkd:
        assert fkd.observation_space == env.observation_space
        assert fkd.reward_space == env.reward_space
        assert fkd.benchmark == env.benchmark


def test_fork_state(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    env.step(0)
    assert env.actions == [0]

    with env.fork() as fkd:
        assert fkd.benchmark == fkd.benchmark
        assert fkd.actions == env.actions


def test_fork_reset(env: LlvmEnv):
    env.reset("cbench-v1/crc32")
    env.step(0)
    env.step(1)
    env.step(2)

    with env.fork() as fkd:
        fkd.step(3)

        assert env.actions == [0, 1, 2]
        assert fkd.actions == [0, 1, 2, 3]

        fkd.reset()
        assert env.actions == [0, 1, 2]
        assert fkd.actions == []


def test_fork_custom_benchmark(env: LlvmEnv):
    benchmark = env.make_benchmark(EXAMPLE_BITCODE_FILE)
    env.reset(benchmark=benchmark)

    def ir(env):
        """Strip the ModuleID line from IR."""
        return "\n".join(env.ir.split("\n")[1:])

    with env.fork() as fkd:
        assert ir(env) == ir(fkd)

        fkd.reset()
        assert ir(env) == ir(fkd)


def test_fork_twice_test(env: LlvmEnv):
    """Test that fork() on a forked environment works."""
    env.reset(benchmark="cbench-v1/crc32")
    with env.fork() as fork_a:
        with fork_a.fork() as fork_b:
            assert env.state == fork_a.state
            assert fork_a.state == fork_b.state


def test_fork_modified_ir_is_the_same(env: LlvmEnv):
    """Test that the IR of a forked environment is the same."""
    env.reset("cbench-v1/crc32")

    # Apply an action that modifies the benchmark.
    _, _, done, info = env.step(env.action_space["-mem2reg"])
    assert not done
    assert not info["action_had_no_effect"]

    with env.fork() as fkd:
        assert "\n".join(env.ir.split("\n")[1:]) == "\n".join(fkd.ir.split("\n")[1:])

        # Apply another action.
        _, _, done, info = env.step(env.action_space["-gvn"])
        _, _, done, info = fkd.step(fkd.action_space["-gvn"])
        assert not done
        assert not info["action_had_no_effect"]

        # Check that IRs are still equivalent.
        assert "\n".join(env.ir.split("\n")[1:]) == "\n".join(fkd.ir.split("\n")[1:])


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
def test_fork_rewards(env: LlvmEnv, reward_space: str):
    """Test that rewards are equal after fork() is called."""
    env.reward_space = reward_space
    env.reset("cbench-v1/dijkstra")

    actions = [
        env.action_space["-mem2reg"],
        env.action_space["-simplifycfg"],
    ]

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

    env.step(env.action_space["-mem2reg"])
    with env.fork() as fkd:
        _, a, _, _ = env.step(env.action_space["-mem2reg"])
        _, b, _, _ = fkd.step(env.action_space["-mem2reg"])
        assert a == b


def test_fork_previous_cost_lazy_reward_update(env: LlvmEnv):
    env.reset("cbench-v1/crc32")

    env.step(env.action_space["-mem2reg"])
    env.reward["IrInstructionCount"]  # noqa
    with env.fork() as fkd:
        env.step(env.action_space["-mem2reg"])
        fkd.step(env.action_space["-mem2reg"])

        assert env.reward["IrInstructionCount"] == fkd.reward["IrInstructionCount"]


def test_forked_service_dies(env: LlvmEnv):
    """Test that if the service dies on a forked environment, each environment
    creates new, independent services.
    """
    with env.fork() as fkd:
        assert env.service == fkd.service
        try:
            fkd.service.connection.close()
        except ServiceError:
            pass  # shutdown() raises service error if in-episode.
        fkd.service.close()

        env.reset()
        fkd.reset()
        assert env.service != fkd.service


if __name__ == "__main__":
    main()
