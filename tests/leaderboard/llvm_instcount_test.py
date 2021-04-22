# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/leaderboard:llvm_instcount."""
from pathlib import Path

import pytest
from absl import flags

from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main

FLAGS = flags.FLAGS

pytest_plugins = ["tests.pytest_plugins.common"]


def null_policy(env) -> None:
    """A policy that does nothing."""
    pass


def test_eval_llvm_instcount_policy():
    set_command_line_flags(["argv0", "--n=1", "--max_benchmarks=1", "--novalidate"])
    with pytest.raises(SystemExit):
        eval_llvm_instcount_policy(null_policy)


def test_eval_llvm_instcount_policy_resume(tmpwd):
    # Run eval on a single benchmark.
    set_command_line_flags(
        [
            "argv0",
            "--n=1",
            "--max_benchmarks=1",
            "--novalidate",
            "--resume",
            "--leaderboard_results=test.csv",
        ]
    )
    with pytest.raises(SystemExit):
        eval_llvm_instcount_policy(null_policy)

    # Check that the log has a single entry (and a header row.)
    assert Path("test.csv").is_file()
    with open("test.csv") as f:
        log = f.read()
    assert len(log.rstrip().split("\n")) == 2
    init_logfile = log

    # Repeat, but for two benchmarks.
    set_command_line_flags(
        [
            "argv0",
            "--n=1",
            "--max_benchmarks=2",
            "--novalidate",
            "--resume",
            "--leaderboard_results=test.csv",
        ]
    )
    with pytest.raises(SystemExit):
        eval_llvm_instcount_policy(null_policy)

    # Check that the log extends the original.
    assert Path("test.csv").is_file()
    with open("test.csv") as f:
        log = f.read()
    assert log.startswith(init_logfile)
    assert len(log.rstrip().split("\n")) == 3
    init_logfile = log

    # Repeat, but for two runs of each benchmark.
    set_command_line_flags(
        [
            "argv0",
            "--n=2",
            "--max_benchmarks=2",
            "--novalidate",
            "--resume",
            "--leaderboard_results=test.csv",
        ]
    )
    with pytest.raises(SystemExit):
        eval_llvm_instcount_policy(null_policy)

    # Check that the log extends the original.
    assert Path("test.csv").is_file()
    with open("test.csv") as f:
        log = f.read()
    assert log.startswith(init_logfile)
    assert len(log.rstrip().split("\n")) == 5


def test_eval_llvm_instcount_policy_invalid_flag():
    set_command_line_flags(["argv0", "--n=-1"])
    with pytest.raises(AssertionError):
        eval_llvm_instcount_policy(null_policy)


if __name__ == "__main__":
    main()
