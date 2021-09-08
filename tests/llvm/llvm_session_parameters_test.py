# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM session parameter handlers."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service import SessionNotFound
from compiler_gym.service.connection import ServiceError
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_send_param_before_reset(env: LlvmEnv):
    """Test that send_params() before reset() raises an error."""
    with pytest.raises(
        SessionNotFound, match=r"Must call reset\(\) before send_params\(\)"
    ):
        env.send_params(("test", "test"))


def test_send_param_unknown_key(env: LlvmEnv):
    """Test that send_params() raises an error when the key is not recognized."""
    env.reset()
    with pytest.raises(ValueError, match="Unknown parameter: unknown.key"):
        env.send_params(("unknown.key", ""))


def test_benchmarks_cache_parameters(env: LlvmEnv):
    env.reset()
    assert int(env.send_param("service.benchmark_cache.get_size_in_bytes", "")) > 0

    # Get the default max size.
    assert env.send_params(("service.benchmark_cache.get_max_size_in_bytes", "")) == [
        str(256 * 1024 * 1024)
    ]
    assert env.send_param(  # Same again but using singular API endpoint.
        "service.benchmark_cache.get_max_size_in_bytes", ""
    ) == str(256 * 1024 * 1024)

    # Set a new max size.
    assert env.send_params(
        ("service.benchmark_cache.set_max_size_in_bytes", "256")
    ) == ["256"]
    assert env.send_params(("service.benchmark_cache.get_max_size_in_bytes", "")) == [
        "256"
    ]


def test_send_param_invalid_reply_count(env: LlvmEnv, mocker):
    """Test that an error is raised when # replies != # params."""
    env.reset()

    mocker.patch.object(env, "service")
    with pytest.raises(
        OSError, match="Sent 1 parameter but received 0 responses from the service"
    ):
        env.send_param("param", "")


def test_benchmarks_cache_parameter_invalid_int_type(env: LlvmEnv):
    env.reset()
    with pytest.raises(ServiceError, match="stoi"):
        env.send_params(("service.benchmark_cache.set_max_size_in_bytes", "not an int"))


@pytest.mark.parametrize("n", [1, 3, 10])
def test_runtime_observation_parameters(env: LlvmEnv, n: int):
    env.observation_space = "Runtime"
    env.reset(benchmark="cbench-v1/qsort")

    assert env.send_param("llvm.set_runtimes_per_observation_count", str(n)) == str(n)
    assert env.send_param("llvm.get_runtimes_per_observation_count", "") == str(n)
    runtimes = env.observation["Runtime"]
    assert len(runtimes) == n

    assert env.observation_space.contains(runtimes)


if __name__ == "__main__":
    main()
