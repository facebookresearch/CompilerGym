# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the GCC CompilerGym environments."""

import gym
import pytest

from compiler_gym.envs.gcc import GccEnv
from tests.pytest_plugins.common import docker_is_available


@pytest.fixture(scope="function")
def env() -> GccEnv:
    """Create a GCC environment."""
    env = gym.make("gcc-v0")
    try:
        yield env
    finally:
        env.close()


# Decorator to skip a test if GCC environment is not supported.
with_gcc_support = pytest.mark.skipif(
    not docker_is_available(), reason="Docker is not available"
)

# Decorator to skip a test if GCC environment is supported.
without_gcc_support = pytest.mark.skipif(
    docker_is_available(), reason="Docker is not available"
)
