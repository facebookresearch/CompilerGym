# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:random_walk."""
import gym
from absl import flags

from examples.random_walk import run_random_walk
from tests.test_main import main


def test_run_random_walk_smoke_test():
    flags.FLAGS(["argv0"])
    env = gym.make("llvm-autophase-ic-v0")
    env.benchmark = "cbench-v1/crc32"
    try:
        run_random_walk(env=env, step_count=5)
    finally:
        env.close()


if __name__ == "__main__":
    main()
