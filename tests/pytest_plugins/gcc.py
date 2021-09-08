# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the GCC CompilerGym environments."""

import gym
import pytest

from compiler_gym.envs.gcc import GccEnv


@pytest.fixture(scope="function")
def env() -> GccEnv:
    """Create a GCC environment."""
    env = gym.make("gcc-v0")
    try:
        yield env
    finally:
        env.close()
