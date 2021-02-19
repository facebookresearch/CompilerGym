# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
from typing import List

import gym
import pytest

import compiler_gym
from compiler_gym.envs import CompilerEnv, CompilerEnvState, llvm
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.service.connection import CompilerGymServiceConnection
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.fixture(scope="function", params=["local", "service"])
def env(request) -> CompilerEnv:
    """Create an LLVM environment."""
    if request.param == "local":
        env = gym.make("llvm-v0")
        env.require_dataset("cBench-v0")
        try:
            yield env
        finally:
            env.close()
    else:
        service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
        env = LlvmEnv(service=service.connection.url, benchmark="foo")
        env.require_dataset("cBench-v0")
        try:
            yield env
        finally:
            env.close()
            service.close()


def test_service_version(env: LlvmEnv):
    assert env.version == compiler_gym.__version__


def test_compiler_version(env: LlvmEnv):
    assert env.compiler_version.startswith("10.0.0")


def test_action_space_names(env: CompilerEnv, action_names: List[str]):
    assert set(env.action_space.names) == set(action_names)


def test_action_spaces_names(env: CompilerEnv):
    assert {a.name for a in env.action_spaces} == {"PassesAll"}


def test_all_flags_are_unique(env: LlvmEnv):
    assert sorted(env.action_space.flags) == sorted(set(env.action_space.flags))


def test_benchmark_names(env: CompilerEnv, benchmark_names: List[str]):
    # Use a subset check as benchmark_names may exclude certain benchmarks for
    # testing.
    assert set(benchmark_names).issubset(set(env.benchmarks))


def test_double_reset(env: CompilerEnv):
    env.reset(benchmark="cBench-v0/crc32")
    env.reset(benchmark="cBench-v0/crc32")
    assert env.in_episode


def test_commandline(env: CompilerEnv):
    env.reset(benchmark="cBench-v0/crc32")
    assert env.commandline() == "opt  input.bc -o output.bc"


def test_uri_substring_candidate_match(env: CompilerEnv):
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="benchmark://cBench-v0/crc3")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="benchmark://cBench-v0/cr")
    assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_uri_substring_candidate_match_infer_protocol(env: CompilerEnv):
    env.reset(benchmark="cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="cBench-v0/crc3")
    assert env.benchmark == "benchmark://cBench-v0/crc32"

    env.reset(benchmark="cBench-v0/cr")
    assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_reset_to_force_benchmark(env: CompilerEnv):
    """Reset that calling reset() with a benchmark forces that benchmark to
    be used for every subsequent episode.
    """
    env.benchmark = None
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    for _ in range(10):
        env.reset()
        assert env.benchmark == "benchmark://cBench-v0/crc32"


def test_unset_forced_benchmark(env: CompilerEnv):
    """Test that setting benchmark to None "unsets" the user benchmark for
    every subsequent episode.
    """
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    env.benchmark = None
    for _ in range(50):
        env.reset()
        if env.benchmark != "benchmark://cBench-v0/crc32":
            break
    else:
        pytest.fail(
            "Improbably selected the same benchmark 50 times! " "Expected random."
        )


def test_change_benchmark_mid_episode(env: LlvmEnv):
    """Test that changing the benchmark while in an episode has no effect until
    the next call to reset()."""
    env.reset(benchmark="benchmark://cBench-v0/crc32")
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    env.benchmark = "benchmark://cBench-v0/dijkstra"
    assert env.benchmark == "benchmark://cBench-v0/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cBench-v0/dijkstra"


def test_set_benchmark_invalid_type(env: LlvmEnv):
    with pytest.raises(TypeError) as ctx:
        env.benchmark = 10
    assert str(ctx.value) == "Unsupported benchmark type: int"


def test_gym_make_kwargs():
    """Test that passing kwargs to gym.make() are forwarded to environment
    constructor.
    """
    env = gym.make(
        "llvm-v0", observation_space="Autophase", reward_space="IrInstructionCount"
    )
    try:
        assert env.observation_space.id == "Autophase"
        assert env.reward_space.id == "IrInstructionCount"
    finally:
        env.close()


def test_connection_dies_default_reward(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cBench-v0/crc32")

    env.reward_space.default_negates_returns = False
    env.reward_space.default_value = 2.5
    env.episode_reward = 10

    env.service.close()
    observation, reward, done, _ = env.step(0)
    assert done

    assert reward == 2.5


def test_connection_dies_default_reward_negated(env: LlvmEnv):
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cBench-v0/crc32")

    env.reward_space.default_negates_returns = True
    env.reward_space.default_value = 2.5
    env.episode_reward = 10

    env.service.close()
    observation, reward, done, _ = env.step(0)
    assert done

    assert reward == -7.5  # negates reward.


def test_state_to_csv_from_csv(env: LlvmEnv):
    env.reset(benchmark="cBench-v0/crc32")
    env.episode_reward = 10

    state = env.state
    assert state.reward == 10
    state_from_csv = CompilerEnvState.from_csv(env.state.to_csv())
    assert state_from_csv.reward == 10
    assert state == state_from_csv


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
    env.benchmark = "cBench-v0/dijkstra"

    action = env.action_space.flags.index("-instcombine")
    env.reset()

    _, reward_a, _, _ = env.step(action)
    assert reward_a, "Sanity check that action produces a reward"

    env.reset()
    _, reward_b, _, _ = env.step(action)
    assert reward_a == reward_b


if __name__ == "__main__":
    main()
