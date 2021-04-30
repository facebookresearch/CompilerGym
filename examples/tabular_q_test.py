# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/bin:tabular_q."""
from absl import flags

from compiler_gym.util.capture_output import capture_output
from examples.tabular_q import main
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def test_run_tabular_q_smoke_test():
    set_command_line_flags(
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


if __name__ == "__main__":
    _test_main()
