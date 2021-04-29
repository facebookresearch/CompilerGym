# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import List

import gym
import pytest

import compiler_gym
from compiler_gym.compiler_env_state import (
    CompilerEnvStateReader,
    CompilerEnvStateWriter,
)
from compiler_gym.envs import CompilerEnv, llvm
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.service.connection import CompilerGymServiceConnection
from compiler_gym.util import debug_util as dbg
from tests.pytest_plugins import llvm as llvm_plugin
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common", "tests.pytest_plugins.llvm"]


@pytest.fixture(scope="function", params=["local", "service"])
def env(request) -> CompilerEnv:
    """Create an LLVM environment."""
    if request.param == "local":
        env = gym.make("llvm-v0")
        try:
            yield env
        finally:
            env.close()
    else:
        service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
        env = LlvmEnv(service=service.connection.url)
        try:
            yield env
        finally:
            env.close()
            service.close()


def test_service_version(env: LlvmEnv):
    assert env.version == compiler_gym.__version__


def test_compiler_version(env: LlvmEnv):
    assert env.compiler_version.startswith("10.0.0")


def test_action_space_names(env: LlvmEnv, action_names: List[str]):
    assert set(env.action_space.names) == set(action_names)


def test_action_spaces_names(env: LlvmEnv):
    assert {a.name for a in env.action_spaces} == {"PassesAll"}


def test_all_flags_are_unique(env: LlvmEnv):
    assert sorted(env.action_space.flags) == sorted(set(env.action_space.flags))


@pytest.mark.parametrize("benchmark_name", llvm_plugin.BENCHMARK_NAMES)
def test_benchmark_names(env: LlvmEnv, benchmark_name: str):
    """Check that all benchmark names can be found in the datasets."""
    assert env.datasets.benchmark(benchmark_name)


def test_double_reset(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    env.reset(benchmark="cbench-v1/crc32")
    assert env.in_episode


def test_commandline_no_actions(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    assert env.commandline() == "opt  input.bc -o output.bc"
    assert env.commandline_to_actions(env.commandline()) == []


def test_commandline(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    env.step(env.action_space.flags.index("-mem2reg"))
    env.step(env.action_space.flags.index("-reg2mem"))
    assert env.commandline() == "opt -mem2reg -reg2mem input.bc -o output.bc"
    assert env.commandline_to_actions(env.commandline()) == [
        env.action_space.flags.index("-mem2reg"),
        env.action_space.flags.index("-reg2mem"),
    ]


def test_connection_dies_default_reward(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cbench-v1/crc32")

    env.reward_space.default_negates_returns = False
    env.reward_space.default_value = 2.5
    env.episode_reward = 10

    env.service.close()
    observation, reward, done, _ = env.step(0)
    assert done

    assert reward == 2.5


def test_connection_dies_default_reward_negated(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cbench-v1/crc32")

    env.reward_space.default_negates_returns = True
    env.reward_space.default_value = 2.5
    env.episode_reward = 10

    env.service.close()
    observation, reward, done, _ = env.step(0)
    assert done

    assert reward == -7.5  # negates reward.


def test_state_serialize_deserialize_equality(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    env.episode_reward = 10

    state = env.state
    assert state.reward == 10

    buf = StringIO()
    CompilerEnvStateWriter(buf).write_state(state)
    buf.seek(0)  # Rewind the buffer for reading.
    state_from_csv = next(iter(CompilerEnvStateReader(buf)))

    assert state_from_csv.reward == 10
    assert state == state_from_csv


def test_apply_state(env: LlvmEnv):
    """Test that apply() on a clean environment produces same state."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cbench-v1/crc32")
    env.step(env.action_space.flags.index("-mem2reg"))

    other = gym.make("llvm-v0", reward_space="IrInstructionCount")
    try:
        other.apply(env.state)

        assert other.state == env.state
    finally:
        other.close()


def test_set_observation_space_from_spec(env: LlvmEnv):
    env.observation_space = env.observation.spaces["Autophase"]
    obs = env.observation_space

    env.observation_space = "Autophase"
    assert env.observation_space == obs


def test_set_reward_space_from_spec(env: LlvmEnv):
    env.reward_space = env.reward.spaces["IrInstructionCount"]
    reward = env.reward_space

    env.reward_space = "IrInstructionCount"
    assert env.reward_space == reward


def test_same_reward_after_reset(env: LlvmEnv):
    """Check that running the same action after calling reset() produces
    same reward.
    """
    env.reward_space = "IrInstructionCount"
    env.benchmark = "cbench-v1/dijkstra"

    action = env.action_space.flags.index("-instcombine")
    env.reset()

    _, reward_a, _, _ = env.step(action)
    assert reward_a, "Sanity check that action produces a reward"

    env.reset()
    _, reward_b, _, _ = env.step(action)
    assert reward_a == reward_b


def test_write_bitcode(env: LlvmEnv, tmpwd: Path):
    env.reset(benchmark="cbench-v1/crc32")
    env.write_bitcode("file.bc")
    assert Path("file.bc").is_file()


def test_write_ir(env: LlvmEnv, tmpwd: Path):
    env.reset(benchmark="cbench-v1/crc32")
    env.write_bitcode("file.ll")
    assert Path("file.ll").is_file()


def test_ir_sha1(env: LlvmEnv, tmpwd: Path):
    env.reset(benchmark="cbench-v1/crc32")
    before = env.ir_sha1

    _, _, done, info = env.step(env.action_space.flags.index("-mem2reg"))
    assert not done, info
    assert not info["action_had_no_effect"], "sanity check failed, action had no effect"

    after = env.ir_sha1
    assert before != after


def test_generate_enum_declarations(env: LlvmEnv):
    assert issubclass(llvm.observation_spaces, Enum)
    assert issubclass(llvm.reward_spaces, Enum)


def test_logging_default_level(env: LlvmEnv):
    assert env.logger.level == dbg.get_logging_level()


def test_step_multiple_actions_list(env: LlvmEnv):
    """Pass a list of actions to step()."""
    env.reset(benchmark="cbench-v1/crc32")
    actions = [
        env.action_space.flags.index("-mem2reg"),
        env.action_space.flags.index("-reg2mem"),
    ]
    _, _, done, _ = env.step(actions)
    assert not done
    assert env.actions == actions


def test_step_multiple_actions_generator(env: LlvmEnv):
    """Pass an iterable of actions to step()."""
    env.reset(benchmark="cbench-v1/crc32")
    actions = (
        env.action_space.flags.index("-mem2reg"),
        env.action_space.flags.index("-reg2mem"),
    )
    _, _, done, _ = env.step(actions)
    assert not done
    assert env.actions == [
        env.action_space.flags.index("-mem2reg"),
        env.action_space.flags.index("-reg2mem"),
    ]


if __name__ == "__main__":
    main()
