# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.fork()."""
import numpy as np

from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


# The number of actions to run before and after calling fork().
PRE_FORK_ACTIONS = 10
POST_FORK_ACTIONS = 10


def test_fuzz(env: LlvmEnv):
    """This test generates a random trajectory and checks that fork() produces
    an equivalent state. It then runs a second trajectory on the two
    environments to check that behavior is consistent across them.
    """
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env.reset()
    print(f"Running fuzz test of environment {env.benchmark}")

    # Take a few warmup steps to get an environment in a random state.
    for _ in range(PRE_FORK_ACTIONS):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:  # Broken episode, restart.
            break
    else:
        # Fork the environment and check that the states are equivalent.
        new_env = env.fork()
        try:
            print(env.state)  # For debugging in case of error.
            assert env.state == new_env.state
            # Check that environment states remain equal if identical
            # subsequent steps are taken.
            for _ in range(POST_FORK_ACTIONS):
                action = env.action_space.sample()
                observation_a, reward_a, done_a, _ = env.step(action)
                observation_b, reward_b, done_b, _ = new_env.step(action)

                print(env.state)  # For debugging in case of error.
                np.testing.assert_array_almost_equal(observation_a, observation_b)
                np.testing.assert_almost_equal(reward_a, reward_b)
                assert done_a == done_b
                if done_a:
                    break  # Broken episode, we're done.
                assert env.state == new_env.state
        finally:
            new_env.close()


if __name__ == "__main__":
    main()
