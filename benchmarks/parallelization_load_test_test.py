# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke test for //benchmarks:parallelization_load_test."""
from pathlib import Path

from absl import flags

from benchmarks.parallelization_load_test import main as load_test
from compiler_gym.util.capture_output import capture_output
from tests.pytest_plugins.common import set_command_line_flags, skip_on_ci
from tests.test_main import main

FLAGS = flags.FLAGS

pytest_plugins = ["tests.pytest_plugins.llvm", "tests.pytest_plugins.common"]


@skip_on_ci
def test_load_test(env, tmpwd):
    del env  # Unused.
    del tmpwd  # Unused.
    set_command_line_flags(
        [
            "arv0",
            "--env=llvm-v0",
            "--benchmark=cbench-v1/crc32",
            "--max_nproc=3",
            "--nproc_increment=1",
            "--num_steps=2",
            "--num_episodes=2",
        ]
    )
    with capture_output() as out:
        load_test(["argv0"])

    assert "Run 1 threaded workers in " in out.stdout
    assert "Run 1 process workers in " in out.stdout
    assert "Run 2 threaded workers in " in out.stdout
    assert "Run 2 process workers in " in out.stdout
    assert "Run 3 threaded workers in " in out.stdout
    assert "Run 3 process workers in " in out.stdout

    assert Path("parallelization_load_test.csv").is_file()


if __name__ == "__main__":
    main()
