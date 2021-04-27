# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.validate()."""
import random

from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.llvm import VALIDATABLE_CBENCH_URIS
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


# The uniform range for trajectory lengths.
RANDOM_TRAJECTORY_LENGTH_RANGE = (1, 50)


def test_fuzz(env: LlvmEnv):
    """This test generates a random trajectory and validates the semantics."""
    benchmark = random.choice(VALIDATABLE_CBENCH_URIS)
    num_actions = random.randint(*RANDOM_TRAJECTORY_LENGTH_RANGE)

    while True:
        env.reset(benchmark=benchmark)
        for _ in range(num_actions):
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break  # Broken trajectory, retry.
        else:
            print(f"Validating state {env.state}")
            assert env.validate() == []
            # Stop the test.
            break


if __name__ == "__main__":
    main()
