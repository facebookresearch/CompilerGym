# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //leaderboard/llvm_codesize:eval_policy."""
import pytest
from absl import flags

from leaderboard.llvm_codesize.random_search.random_search import (
    eval_policy,
    random_search,
)
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def test_random_search():
    FLAGS.unparse_flags()
    FLAGS(
        [
            "argv0",
            "--n=1",
            "--max_benchmarks=1",
            "--search_time=1",
            "--nproc=1",
            "--patience_ratio=0.1",
            "--novalidate",
        ]
    )
    with pytest.raises(SystemExit):
        eval_policy(random_search)


if __name__ == "__main__":
    _test_main()
