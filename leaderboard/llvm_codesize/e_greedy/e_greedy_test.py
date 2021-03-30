# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //leaderboard/llvm_codesize/e_greedy."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from absl import flags

from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_codesize import eval_llvm_codesize_policy
from leaderboard.llvm_codesize.e_greedy.e_greedy import (
    e_greedy_search,
    select_best_action,
)
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_random_search():
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
        eval_llvm_codesize_policy(e_greedy_search)


def test_select_best_action_closed_environment(env: LlvmEnv):
    """Test that select_best_action() recovers from an environment whose service
    has closed."""
    env.reward_space = "IrInstructionCount"
    env.reset(benchmark="cBench-v1/crc32")
    with ThreadPoolExecutor() as executor:
        best_a = select_best_action(env, executor)
        env.close()
        best_b = select_best_action(env, executor)
        assert best_a == best_b


if __name__ == "__main__":
    _test_main()
