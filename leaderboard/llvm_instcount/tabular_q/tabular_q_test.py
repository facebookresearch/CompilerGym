# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //leaderboard/llvm_instcount/tabular_q_eval."""
import pytest
from absl import flags

from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from leaderboard.llvm_instcount.tabular_q.tabular_q_eval import train_and_run
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_tabular_q():
    FLAGS.unparse_flags()
    FLAGS(
        [
            "argv0",
            "--n=1",
            "--max_benchmarks=1",
            "--nproc=1",
            "--novalidate",
        ]
    )
    with pytest.raises(SystemExit):
        eval_llvm_instcount_policy(train_and_run)


if __name__ == "__main__":
    _test_main()
