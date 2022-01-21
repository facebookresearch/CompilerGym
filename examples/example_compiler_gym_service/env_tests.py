# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the example CompilerGym service."""
import socket
import subprocess
from pathlib import Path
from time import sleep

import gym
import numpy as np
import pytest
from flaky import flaky

import examples.example_compiler_gym_service as example
from compiler_gym.envs import CompilerEnv
from compiler_gym.service import SessionNotFound
from compiler_gym.spaces import Box, NamedDiscrete, Scalar, Sequence
from compiler_gym.util.commands import Popen
from tests.test_main import main

# Given that the C++ and Python service implementations have identical
# featuresets, we can parameterize the tests and run them against both backends.
EXAMPLE_ENVIRONMENTS = ["example-cc-v0", "example-py-v0"]


@pytest.fixture(scope="function", params=EXAMPLE_ENVIRONMENTS)
def env(request) -> CompilerEnv:
    """Text fixture that yields an environment."""
    with gym.make(request.param) as env:
        yield env


@pytest.fixture(
    scope="module",
    params=[example.EXAMPLE_CC_SERVICE_BINARY, example.EXAMPLE_PY_SERVICE_BINARY],
    ids=["example-cc-v0", "example-py-v0"],
)
def bin(request) -> Path:
    yield request.param


def test_invalid_arguments(bin: Path):
    """Test that running the binary with unrecognized arguments is an error."""

    def run(cmd):
        with Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        ) as p:
            stdout, stderr = p.communicate(timeout=10)
            return p.returncode, stdout, stderr

    returncode, _, stderr = run([str(bin), "foobar"])
    assert "ERROR:" in stderr
    assert "'foobar'" in stderr
    assert returncode == 1

    returncode, _, stderr = run([str(bin), "--foobar"])
    # C++ and python flag parsing library emit slightly different error
    # messages.
    assert "ERROR:" in stderr or "FATAL" in stderr
    assert "'foobar'" in stderr
    assert returncode == 1


def test_versions(env: CompilerEnv):
    """Tests the GetVersion() RPC endpoint."""
    assert env.compiler_version == "1.0.0"


def test_action_space(env: CompilerEnv):
    """Test that the environment reports the service's action spaces."""
    assert env.action_spaces == [
        NamedDiscrete(
            name="default",
            items=["a", "b", "c"],
        )
    ]


def test_observation_spaces(env: CompilerEnv):
    """Test that the environment reports the service's observation spaces."""
    env.reset()
    assert env.observation.spaces.keys() == {"ir", "features", "runtime"}
    assert env.observation.spaces["ir"].space == Sequence(
        name="test", size_range=(0, None), dtype=str, opaque_data_format=""
    )
    assert env.observation.spaces["features"].space == Box(
        name="test", shape=(3,), low=-100, high=100, dtype=int
    )
    assert env.observation.spaces["runtime"].space == Scalar(
        name="test", min=0, max=np.inf, dtype=float
    )


def test_reward_spaces(env: CompilerEnv):
    """Test that the environment reports the service's reward spaces."""
    env.reset()
    assert env.reward.spaces.keys() == {"runtime"}


def test_step_before_reset(env: CompilerEnv):
    """Taking a step() before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        env.step(0)


def test_observation_before_reset(env: CompilerEnv):
    """Taking an observation before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.observation["ir"]


def test_reward_before_reset(env: CompilerEnv):
    """Taking a reward before reset() is illegal."""
    with pytest.raises(SessionNotFound, match=r"Must call reset\(\) before step\(\)"):
        _ = env.reward["runtime"]


def test_reset_invalid_benchmark(env: CompilerEnv):
    """Test requesting a specific benchmark."""
    with pytest.raises(LookupError) as ctx:
        env.reset(benchmark="example-v0/foobar")
    assert str(ctx.value) == "Unknown program name"


def test_invalid_observation_space(env: CompilerEnv):
    """Test error handling with invalid observation space."""
    with pytest.raises(LookupError):
        env.observation_space = 100


def test_invalid_reward_space(env: CompilerEnv):
    """Test error handling with invalid reward space."""
    with pytest.raises(LookupError):
        env.reward_space = 100


def test_double_reset(env: CompilerEnv):
    """Test that reset() can be called twice."""
    env.reset()
    assert env.in_episode
    env.reset()
    assert env.in_episode


def test_double_reset_with_step(env: CompilerEnv):
    """Test that reset() can be called twice with a step."""
    env.reset()
    assert env.in_episode
    _, _, done, info = env.step(env.action_space.sample())
    assert not done, info
    env.reset()
    _, _, done, info = env.step(env.action_space.sample())
    assert not done, info
    assert env.in_episode


def test_Step_out_of_range(env: CompilerEnv):
    """Test error handling with an invalid action."""
    env.reset()
    with pytest.raises(ValueError) as ctx:
        env.step(100)
    assert str(ctx.value) == "Out-of-range"


def test_default_ir_observation(env: CompilerEnv):
    """Test default observation space."""
    env.observation_space = "ir"
    observation = env.reset()
    assert observation == "Hello, world!"

    observation, reward, done, info = env.step(0)
    assert observation == "Hello, world!"
    assert reward is None
    assert not done


def test_default_features_observation(env: CompilerEnv):
    """Test default observation space."""
    env.observation_space = "features"
    observation = env.reset()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (3,)
    assert observation.dtype == np.int64
    assert observation.tolist() == [0, 0, 0]


def test_default_reward(env: CompilerEnv):
    """Test default reward space."""
    env.reward_space = "runtime"
    env.reset()
    observation, reward, done, info = env.step(0)
    assert observation is None
    assert reward == 0
    assert not done


def test_observations(env: CompilerEnv):
    """Test observation spaces."""
    env.reset()
    assert env.observation["ir"] == "Hello, world!"
    np.testing.assert_array_equal(env.observation["features"], [0, 0, 0])


def test_rewards(env: CompilerEnv):
    """Test reward spaces."""
    env.reset()
    assert env.reward["runtime"] == 0


def test_benchmarks(env: CompilerEnv):
    assert list(env.datasets.benchmark_uris()) == [
        "benchmark://example-v0/foo",
        "benchmark://example-v0/bar",
    ]


def test_fork(env: CompilerEnv):
    env.reset()
    env.step(0)
    env.step(1)
    other_env = env.fork()
    try:
        assert env.benchmark == other_env.benchmark
        assert other_env.actions == [0, 1]
    finally:
        other_env.close()


@flaky  # Timeout-based test.
def test_force_working_dir(bin: Path, tmpdir):
    """Test that expected files are generated in the working directory."""
    tmpdir = Path(tmpdir) / "subdir"
    with Popen([str(bin), "--working_dir", str(tmpdir)]):
        for _ in range(10):
            sleep(0.5)
            if (tmpdir / "pid.txt").is_file() and (tmpdir / "port.txt").is_file():
                break
        else:
            pytest.fail(f"PID file not found in {tmpdir}: {list(tmpdir.iterdir())}")


def unsafe_select_unused_port() -> int:
    """Try and select an unused port that on the local system.

    There is nothing to prevent the port number returned by this function from
    being claimed by another process or thread, so it is liable to race conditions
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def port_is_free(port: int) -> bool:
    """Determine if a port is in use"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


@flaky  # Unsafe free port allocation
def test_force_port(bin: Path, tmpdir):
    """Test that a forced --port value is respected."""
    port = unsafe_select_unused_port()
    assert port_is_free(port)  # Sanity check

    tmpdir = Path(tmpdir)
    with Popen([str(bin), "--port", str(port), "--working_dir", str(tmpdir)]):
        for _ in range(10):
            sleep(0.5)
            if (tmpdir / "pid.txt").is_file() and (tmpdir / "port.txt").is_file():
                break
        else:
            pytest.fail(f"PID file not found in {tmpdir}: {list(tmpdir.iterdir())}")

        with open(tmpdir / "port.txt") as f:
            actual_port = int(f.read())

        assert actual_port == port
        assert not port_is_free(actual_port)


if __name__ == "__main__":
    main()
