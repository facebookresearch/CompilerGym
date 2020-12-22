# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""

from time import time

import gym
import numpy as np

from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


FUZZ_TIME_SECONDS = 2


def make_env(benchmark_name: str) -> CompilerEnv:
    """Construct an environment for testing."""
    env = gym.make("llvm-v0")
    env.eager_reward_space = "IrInstructionCount"
    env.eager_observation_space = "Autophase"
    env.reset(benchmark=benchmark_name)
    return env


def test_benchmark_random_actions(benchmark_name: str):
    """Run randomly selected actions on a benchmark until a minimum amount of time has elapsed."""
    env = make_env(benchmark_name)

    try:
        # Take a random step until a predetermined amount of time has elapsed.
        end_time = time() + FUZZ_TIME_SECONDS
        while time() < end_time:
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                assert observation is None
                assert reward is None
                env = make_env(benchmark_name)
            else:
                assert isinstance(observation, np.ndarray)
                assert observation.shape == (56,)
                assert isinstance(reward, float)
    finally:
        env.close()


if __name__ == "__main__":
    main()
