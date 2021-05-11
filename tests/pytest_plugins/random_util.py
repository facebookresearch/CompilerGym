# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for random testing."""
import random
from time import time
from typing import List, Tuple

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ObservationType


def apply_random_trajectory(
    env: CompilerEnv,
    random_trajectory_length_range=(1, 50),
    timeout: int = 0,
) -> List[Tuple[int, ObservationType, float, bool]]:
    """Evaluate and return a random trajectory."""
    end_time = time() + timeout
    num_actions = random.randint(*random_trajectory_length_range)
    trajectory = []
    for _ in range(num_actions):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        if done:
            break  # Broken trajectory.
        trajectory.append((action, observation, reward, done))
        if timeout and time() > end_time:
            break

    return trajectory
