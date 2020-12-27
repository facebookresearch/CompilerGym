# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""

import gym
import pytest

from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


@pytest.fixture(scope="module")
def env() -> CompilerEnv:
    # Use the same environment for all tests in this shard.
    env = gym.make("llvm-v0")
    env.require_dataset("cBench-v0")
    try:
        yield env
    finally:
        env.close()


def test_step(env: CompilerEnv, observation_space: str, reward_space: str):
    """Request every combination of observation and reward in a fresh environment."""
    env.reward_space = None
    env.observation_space = None
    env.reset(benchmark="cBench-v0/crc32")

    observation = env.observation[observation_space]
    assert observation is not None

    reward = env.reward[reward_space]
    assert reward is not None


if __name__ == "__main__":
    main()
