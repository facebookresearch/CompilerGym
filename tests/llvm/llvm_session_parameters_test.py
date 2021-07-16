# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM session parameter handlers."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service import SessionNotFound
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


if __name__ == "__main__":
    main()
