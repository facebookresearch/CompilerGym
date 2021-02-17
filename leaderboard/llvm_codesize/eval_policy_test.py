# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //leaderboard/llvm_codesize:eval_policy."""
import pytest
from absl import flags

from leaderboard.llvm_codesize.eval_policy import eval_policy
from tests.test_main import main

FLAGS = flags.FLAGS


def null_policy(env) -> None:
    """A policy that does nothing."""
    pass


def test_eval_policy():
    FLAGS.unparse_flags()
    FLAGS(["argv0", "--n=1", "--max_benchmarks=1", "--novalidate"])
    with pytest.raises(SystemExit):
        eval_policy(null_policy)


def test_eval_policy_invalid_flag():
    FLAGS.unparse_flags()
    FLAGS(["argv0", "--n=-1"])
    with pytest.raises(AssertionError):
        eval_policy(null_policy)


if __name__ == "__main__":
    main()
