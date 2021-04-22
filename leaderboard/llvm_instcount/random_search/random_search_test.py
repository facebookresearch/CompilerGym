# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //leaderboard/llvm_instcount/random_search."""
import pytest

from leaderboard.llvm_instcount.random_search.random_search import (
    eval_llvm_instcount_policy,
    random_search,
)
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main as _test_main


def test_random_search():
    set_command_line_flags(
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
        eval_llvm_instcount_policy(random_search)


if __name__ == "__main__":
    _test_main()
