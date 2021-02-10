# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""

from time import time

import gym
import numpy as np

from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_DIM
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


FUZZ_TIME_SECONDS = 2


def test_benchmark_random_actions(benchmark_name: str):
    """Run randomly selected actions on a benchmark until a minimum amount of time has elapsed."""
    env = gym.make(
        "llvm-v0",
        reward_space="IrInstructionCount",
        observation_space="Autophase",
        benchmark=benchmark_name,
    )
    env.reset()

    try:
        # Take a random step until a predetermined amount of time has elapsed.
        end_time = time() + FUZZ_TIME_SECONDS
        while time() < end_time:
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                # Default-value for observation is an array of zeros.
                np.testing.assert_array_equal(
                    observation, np.zeros((AUTOPHASE_FEATURE_DIM,))
                )
                assert isinstance(reward, float)
                env = gym.make(
                    "llvm-v0",
                    reward_space="IrInstructionCount",
                    observation_space="Autophase",
                    benchmark=benchmark_name,
                )
                env.reset()
            else:
                assert isinstance(observation, np.ndarray)
                assert observation.shape == (AUTOPHASE_FEATURE_DIM,)
                assert isinstance(reward, float)
    finally:
        env.close()


if __name__ == "__main__":
    main()
