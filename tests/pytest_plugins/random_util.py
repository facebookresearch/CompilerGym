# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for random testing."""
import random
from typing import List, Tuple

from compiler_gym.envs import CompilerEnv
from compiler_gym.service import observation_t


def apply_random_trajectory(
    env: CompilerEnv, random_trajectory_length_range=(1, 50)
) -> List[Tuple[int, observation_t, float, bool]]:
    """Evaluate and return a random trajectory."""
    num_actions = random.randint(*random_trajectory_length_range)
    trajectory = []
    for _ in range(num_actions):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            break  # Broken trajectory.
        trajectory.append((action, observation, reward, done))

    return trajectory
