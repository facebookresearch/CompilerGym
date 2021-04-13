# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate tabular_q policy for leaderboard."""
import os
import sys
from typing import Dict

from absl import app, flags

from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../../examples")
from tabular_q import (  # noqa pylint: disable=wrong-import-position
    StateActionTuple,
    rollout,
    train,
)

FLAGS = flags.FLAGS


def train_and_run(env: LlvmEnv) -> None:
    """ Run tabular Q learning on an environment """
    FLAGS.log_every = 0  # Disable printing to stdout

    q_table: Dict[StateActionTuple, float] = {}
    env.observation_space = "Autophase"
    training_env = env.fork()
    train(q_table, training_env)
    training_env.close()
    rollout(q_table, env, printout=False)


if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(train_and_run))
