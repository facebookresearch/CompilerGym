# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/envs."""
import logging

import gym
import pytest

from compiler_gym.envs import CompilerEnv, llvm
from compiler_gym.service.connection import CompilerGymServiceConnection
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_benchmark_constructor_arg(env: CompilerEnv):
    env.close()  # Fixture only required to pull in dataset.

    env = gym.make("llvm-v0", benchmark="cbench-v1/dijkstra")
    try:
        assert env.benchmark == "benchmark://cbench-v1/dijkstra"
    finally:
        env.close()


def test_benchmark_setter(env: CompilerEnv):
    env.benchmark = "benchmark://cbench-v1/dijkstra"
    assert env.benchmark != "benchmark://cbench-v1/dijkstra"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_benchmark_set_in_reset(env: CompilerEnv):
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"


def test_logger_forced():
    logger = logging.getLogger("test_logger")
    env_a = gym.make("llvm-v0")
    env_b = gym.make("llvm-v0", logger=logger)
    try:
        assert env_a.logger != logger
        assert env_b.logger == logger
    finally:
        env_a.close()
        env_b.close()


def test_uri_substring_no_match(env: CompilerEnv):
    env.reset(benchmark="benchmark://cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"

    with pytest.raises(LookupError):
        env.reset(benchmark="benchmark://cbench-v1/crc3")

    with pytest.raises(LookupError):
        env.reset(benchmark="benchmark://cbench-v1/cr")


def test_uri_substring_candidate_no_match_infer_protocol(env: CompilerEnv):
    env.reset(benchmark="cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"

    with pytest.raises(LookupError):
        env.reset(benchmark="cbench-v1/crc3")

    with pytest.raises(LookupError):
        env.reset(benchmark="cbench-v1/cr")


def test_reset_to_force_benchmark(env: CompilerEnv):
    """Reset that calling reset() with a benchmark forces that benchmark to
    be used for every subsequent episode.
    """
    env.reset(benchmark="benchmark://cbench-v1/crc32")
    assert env.benchmark == "benchmark://cbench-v1/crc32"
    for _ in range(10):
        env.reset()
        assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_unset_forced_benchmark(env: CompilerEnv):
    """Test that setting benchmark "unsets" the previous benchmark."""
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")

    with pytest.warns(
        UserWarning,
        match=r"Changing the benchmark has no effect until reset\(\) is called",
    ):
        env.benchmark = "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_change_benchmark_mid_episode(env: CompilerEnv):
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


def test_set_benchmark_invalid_type(env: CompilerEnv):
    with pytest.raises(TypeError) as ctx:
        env.benchmark = 10
    assert str(ctx.value) == "Expected a Benchmark or str, received: 'int'"


def test_gym_make_kwargs():
    """Test that passing kwargs to gym.make() are forwarded to environment
    constructor.
    """
    env = gym.make(
        "llvm-v0", observation_space="Autophase", reward_space="IrInstructionCount"
    )
    try:
        assert env.observation_space_spec.id == "Autophase"
        assert env.reward_space.id == "IrInstructionCount"
    finally:
        env.close()


def test_step_session_id_not_found(env: CompilerEnv):
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
def remote_env() -> CompilerEnv:
    """A test fixture that yields a connection to a remote service."""
    service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
    env = CompilerEnv(service=service.connection.url)
    try:
        yield env
    finally:
        env.close()
        service.close()


def test_base_class_has_no_benchmark(remote_env: CompilerEnv):
    """Test that when instantiating the base CompilerEnv class there are no
    datasets available.
    """
    assert remote_env.benchmark is None
    with pytest.raises(TypeError, match="No benchmark set"):
        remote_env.reset()


if __name__ == "__main__":
    main()
