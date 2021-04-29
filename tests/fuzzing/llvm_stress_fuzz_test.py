# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test LLVM backend using llvm-stress."""
import random
import subprocess

from compiler_gym.envs import LlvmEnv
from compiler_gym.service.proto import Benchmark, File
from compiler_gym.third_party import llvm
from tests.pytest_plugins.random_util import apply_random_trajectory
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# The uniform range for trajectory lengths.
RANDOM_TRAJECTORY_LENGTH_RANGE = (1, 10)


def test_fuzz(env: LlvmEnv, observation_space: str, reward_space: str):
    """This test produces a random trajectory using a program generated using
    llvm-stress.
    """
    seed = random.randint(0, 2 << 31)
    llvm_ir = subprocess.check_output([str(llvm.llvm_stress_path()), f"--seed={seed}"])
    print(f"llvm-stress --seed={seed}")  # For debugging in case of failure.
    env.benchmark = Benchmark(uri="stress", program=File(contents=llvm_ir))

    env.observation_space = observation_space
    env.reward_space = reward_space

    env.reset()
    apply_random_trajectory(
        env,
        random_trajectory_length_range=RANDOM_TRAJECTORY_LENGTH_RANGE,
        timeout=10,
    )
    print(env.state)  # For debugging in case of failure.


if __name__ == "__main__":
    main()
