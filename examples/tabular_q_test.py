# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/bin:tabular_q."""
from absl import flags
from tabular_q import main

from compiler_gym.util.capture_output import capture_output

FLAGS = flags.FLAGS


def test_run_tabular_q_smoke_test():
    FLAGS.unparse_flags()
    FLAGS(
        [
            "argv0",
            "--episode_length=5",
            "--episodes=10",
            "--log_every=2",
            "--benchmark=cbench-v1/crc32",
        ]
    )
    with capture_output() as out:
        main(["argv0"])

    assert "Resulting sequence" in out.stdout
