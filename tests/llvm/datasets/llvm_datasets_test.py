# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the LLVM datasets."""
import gym

import compiler_gym.envs.llvm  # noqa register environments
from tests.test_main import main


def test_default_dataset_list():
    env = gym.make("llvm-v0")
    try:
        assert list(d.name for d in env.datasets) == [
            "benchmark://cbench-v1",
            "benchmark://anghabench-v1",
            "benchmark://blas-v0",
            "benchmark://clgen-v0",
            "benchmark://github-v0",
            "benchmark://linux-v0",
            "benchmark://mibench-v0",
            "benchmark://npb-v0",
            "benchmark://opencv-v0",
            "benchmark://poj104-v1",
            "benchmark://tensorflow-v0",
            "generator://csmith-v0",
            "generator://llvm-stress-v0",
        ]
    finally:
        env.close()


if __name__ == "__main__":
    main()
