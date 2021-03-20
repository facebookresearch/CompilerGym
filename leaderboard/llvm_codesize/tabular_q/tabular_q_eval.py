# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate tabular_q policy for leaderboard."""
from typing import Dict

from compiler_gym.envs import LlvmEnv
from examples.tabular_q import StateActionTuple, rollout, train_q
from leaderboard.llvm_codesize.eval_policy import eval_policy


def train_and_run(env: LlvmEnv) -> None:
    """ Run tabular Q learning on an environment """
    q_table: Dict[StateActionTuple, float] = {}
    train_q(q_table, env.fork())
    rollout(q_table, env, True)


if __name__ == "__main__":
    eval_policy(train_and_run)
