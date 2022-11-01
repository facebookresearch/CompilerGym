# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the example CompilerGym service."""
import os
import socket
import subprocess
import sys
from getpass import getuser
from pathlib import Path
from time import sleep
from typing import Iterable, List, Optional

import gym
import numpy as np
import pytest
from flaky import flaky

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs import CompilerEnv
from compiler_gym.service import SessionNotFound
from compiler_gym.spaces import Box, NamedDiscrete, Reward, Scalar, Sequence
from compiler_gym.util import debug_util as dbg
from compiler_gym.util.commands import Popen
from compiler_gym.util.registration import register

EXAMPLE_PY_SERVICE_BINARY: Path = Path(
    "example_compiler_gym_service/service_py/example_service.py"
)
assert EXAMPLE_PY_SERVICE_BINARY.is_file(), "Service script not found"


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_runtime = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]

        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


class ExampleDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://example-compiler-v0",
            license="MIT",
            description="An example dataset",
        )
        self._benchmarks = {
            "/foo": Benchmark.from_file_contents(
                "benchmark://example-compiler-v0/foo", "Ir data".encode("utf-8")
            ),
            "/bar": Benchmark.from_file_contents(
                "benchmark://example-compiler-v0/bar", "Ir data".encode("utf-8")
            ),
        }

    def benchmark_uris(self) -> Iterable[str]:
        yield from (
            f"benchmark://example-compiler-v0{k}" for k in self._benchmarks.keys()
        )

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown program name")


# Registexample-compiler-v0ironment for use with gym.make(...).
register(
    id="example-compiler-v0",
    entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
    kwargs={
        "service": EXAMPLE_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
        "datasets": [ExampleDataset()],
    },
)

# Given that the C++ and Python service implementations have identical
# featuresets, we can parameterize the tests and run them against both backends.
EXAMPLE_ENVIRONMENTS = ["example-compiler-v0"]


@pytest.fixture(scope="function", params=EXAMPLE_ENVIRONMENTS)
def env(request) -> CompilerEnv:
    """Text fixture that yields an environment."""
    with gym.make(request.param) as env:
        yield env


@pytest.fixture(
    scope="module",
    params=[
        EXAMPLE_PY_SERVICE_BINARY,
    ],
    ids=["example-compiler-v0"],
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

    ir_space = env.observation.spaces["ir"]
    assert isinstance(ir_space.space, Sequence)
    assert ir_space.space.dtype == str
    assert ir_space.space.size_range == (0, np.iinfo(np.int64).max)

    feature_space = env.observation.spaces["features"].space
    assert isinstance(feature_space, Box)
    assert feature_space.shape == (3,)
    assert np.all(feature_space.low == [-100, -100, -100])
    assert np.all(feature_space.high == [100, 100, 100])
    assert feature_space.dtype == int

    runtime_space = env.observation.spaces["runtime"].space
    assert isinstance(runtime_space, Scalar)
    assert runtime_space.min == 0
    assert runtime_space.max == np.inf
    assert runtime_space.dtype == float


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
        env.reset(benchmark="example-compiler-v0/foobar")
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
        "benchmark://example-compiler-v0/foo",
        "benchmark://example-compiler-v0/bar",
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


# Copied from CompilerGym/tests/test_main.py because there were errors in trying to import it here
def main(extra_pytest_args: Optional[List[str]] = None, debug_level: int = 1):
    dbg.set_debug_level(debug_level)

    # Keep test data isolated from user data.
    os.environ[
        "COMPILER_GYM_SITE_DATA"
    ] = f"/tmp/compiler_gym_{getuser()}/tests/site_data"
    os.environ["COMPILER_GYM_CACHE"] = f"/tmp/compiler_gym_{getuser()}/tests/cache"

    pytest_args = sys.argv + [
        # Run pytest verbosely to print out test names to provide context in
        # case of failures.
        "-vv",
        # Disable "Module already imported" warnings. See:
        # https://docs.pytest.org/en/latest/how-to/usage.html#calling-pytest-from-python-code
        "-W",
        "ignore:Module already imported:pytest.PytestWarning",
        # Disable noisy "Flaky tests passed" messages.
        "--no-success-flaky-report",
    ]
    # Support for sharding. If a py_test target has the shard_count attribute
    # set (in the range [1,50]), then the pytest-shard module is used to divide
    # the tests among the shards. See https://pypi.org/project/pytest-shard/
    sharded_test = os.environ.get("TEST_TOTAL_SHARDS")
    if sharded_test:
        num_shards = int(os.environ["TEST_TOTAL_SHARDS"])
        shard_index = int(os.environ["TEST_SHARD_INDEX"])
        pytest_args += [f"--shard-id={shard_index}", f"--num-shards={num_shards}"]
    else:
        pytest_args += ["-p", "no:pytest-shard"]

    pytest_args += extra_pytest_args or []

    returncode = pytest.main(pytest_args)

    # By default pytest will fail with an error if no tests are collected.
    # Disable that behavior here (with a warning) since there are legitimate
    # cases where we may want to run a test file with no tests in it. For
    # example, when running on a continuous integration service where all the
    # tests are marked with the @skip_on_ci decorator.
    if returncode == pytest.ExitCode.NO_TESTS_COLLECTED.value:
        print(
            "WARNING: The test suite was empty. Is that intended?",
            file=sys.stderr,
        )
        returncode = 0

    sys.exit(returncode)


if __name__ == "__main__":
    main(
        extra_pytest_args=[
            "-W",
            "ignore::UserWarning",
        ]
    )
