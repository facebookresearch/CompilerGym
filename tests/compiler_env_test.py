# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/envs."""
import gym
import pytest
from flaky import flaky

from compiler_gym.envs import llvm
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service.connection import CompilerGymServiceConnection
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_benchmark_constructor_arg(env: LlvmEnv):
    env.close()  # Fixture only required to pull in dataset.

    with gym.make("llvm-v0", benchmark="cbench-v1/dijkstra") as env:
        assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_benchmark_setter(env: LlvmEnv):
    env.benchmark = "benchmark://cbench-v1/dijkstra"
    assert env.benchmark != "benchmark://cbench-v1/dijkstra"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_benchmark_set_in_reset(env: LlvmEnv):
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_logger_is_deprecated(env: LlvmEnv):
    with pytest.deprecated_call(
        match="The `CompilerEnv.logger` attribute is deprecated"
    ):
        env.logger


def test_uri_substring_no_match(env: LlvmEnv):
    env.reset(benchmark="benchmark://cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"

    with pytest.raises(LookupError):
        env.reset(benchmark="benchmark://cbench-v1/crc3")

    with pytest.raises(LookupError):
        env.reset(benchmark="benchmark://cbench-v1/cr")


def test_uri_substring_candidate_no_match_infer_scheme(env: LlvmEnv):
    env.reset(benchmark="cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"

    with pytest.raises(LookupError):
        env.reset(benchmark="cbench-v1/crc3")

    with pytest.raises(LookupError):
        env.reset(benchmark="cbench-v1/cr")


def test_reset_to_force_benchmark(env: LlvmEnv):
    """Reset that calling reset() with a benchmark forces that benchmark to
    be used for every subsequent episode.
    """
    env.reset(benchmark="benchmark://cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    for _ in range(10):
        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_unset_forced_benchmark(env: LlvmEnv):
    """Test that setting benchmark "unsets" the previous benchmark."""
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")

    with pytest.warns(
        UserWarning,
        match=r"Changing the benchmark has no effect until reset\(\) is called",
    ):
        env.benchmark = "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_change_benchmark_mid_episode(env: LlvmEnv):
    """Test that changing the benchmark while in an episode has no effect until
    the next call to reset()."""
    env.reset(benchmark="benchmark://cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    with pytest.warns(
        UserWarning,
        match=r"Changing the benchmark has no effect until reset\(\) is called",
    ):
        env.benchmark = "benchmark://cbench-v1/dijkstra"
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_set_benchmark_invalid_type(env: LlvmEnv):
    with pytest.raises(TypeError) as ctx:
        env.benchmark = 10
    assert str(ctx.value) == "Expected a Benchmark or str, received: 'int'"


def test_gym_make_kwargs():
    """Test that passing kwargs to gym.make() are forwarded to environment
    constructor.
    """
    with gym.make(
        "llvm-v0", observation_space="Autophase", reward_space="IrInstructionCount"
    ) as env:
        assert env.observation_space_spec.id == "Autophase"
        assert env.reward_space.id == "IrInstructionCount"


def test_step_session_id_not_found(env: LlvmEnv):
    """Test that step() recovers gracefully from an unknown session error from
    the service."""
    env._session_id = 15  # pylint: disable=protected-access
    observation, reward, done, info = env.step(0)
    assert done
    assert info["error_details"] == "Session not found: 15"
    assert observation is None
    assert reward is None
    assert not env.in_episode


@pytest.fixture(scope="function")
def remote_env() -> LlvmEnv:
    """A test fixture that yields a connection to a remote service."""
    service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
    try:
        with LlvmEnv(service=service.connection.url) as env:
            yield env
    finally:
        service.close()


@flaky  # step() can fail.
def test_switch_default_reward_space_in_episode(env: LlvmEnv):
    """Test that switching reward space during an episode resets the cumulative
    episode reward.
    """
    env.reward_space = None

    env.reset()
    _, _, done, info = env.step(0)
    assert not done, info
    assert env.episode_reward is None

    env.reward_space = "IrInstructionCount"
    assert env.episode_reward == 0

    _, _, done, info = env.step(0)
    assert not done, info
    assert env.episode_reward is not None


@flaky  # step() can fail.
def test_set_same_default_reward_space_in_episode(env: LlvmEnv):
    """Test that setting the reward space during an episode does not reset the
    cumulative episode reward if the reward space is unchanged.
    """
    env.reward_space = "IrInstructionCount"

    env.reset()

    env.episode_reward = 10

    # No change to the reward space.
    env.reward_space = "IrInstructionCount"
    assert env.episode_reward == 10

    # Change in reward space.
    env.reward_space = "IrInstructionCountOz"
    assert env.episode_reward == 0


if __name__ == "__main__":
    main()
