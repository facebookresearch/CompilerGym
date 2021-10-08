# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/bin:actor_critic."""
import sys

from absl import flags
from actor_critic import main

from compiler_gym.util.capture_output import capture_output

FLAGS = flags.FLAGS


def test_run_actor_critic_smoke_test():
    flags = [
        "argv0",
        "--seed=0",
        "--episode_len=2",
        "--episodes=10",
        "--log_interval=5",
        "--benchmark=cbench-v1/crc32",
    ]
    sys.argv = flags
    FLAGS.unparse_flags()
    FLAGS(flags)
    with capture_output() as out:
        main(["argv0"])

    assert "Final performance (avg reward)" in out.stdout
