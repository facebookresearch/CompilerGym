# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.fork()."""
from time import time

from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


FUZZ_TIME_SECONDS = 30
A_FEW_RANDOM_ACTIONS = 5


def test_fork_state_fuzz_test(env: LlvmEnv):
    """Run random episodes and check that fork() produces equivalent state."""
    end_time = time() + FUZZ_TIME_SECONDS
    while time() < end_time:
        env.reset(benchmark="cBench-v0/dijkstra")

        # Take a few warmup steps to get an environment in a random state.
        for _ in range(A_FEW_RANDOM_ACTIONS):
            _, _, done, _ = env.step(env.action_space.sample())
            if done:  # Broken episode, restart.
                break
        else:
            # Fork the environment and check that the states are equivalent.
            new_env = env.fork()
            try:
                assert env.state == new_env.state
                # Check that environment states remain equal if identical
                # subsequent steps are taken.
                for _ in range(A_FEW_RANDOM_ACTIONS):
                    action = env.action_space.sample()
                    _, _, done_a, _ = env.step(action)
                    _, _, done_b, _ = new_env.step(action)
                    assert done_a == done_b
                    if done_a:  # Broken episode, restart.
                        break
                    assert env.state == new_env.state
            finally:
                new_env.close()


if __name__ == "__main__":
    main()
