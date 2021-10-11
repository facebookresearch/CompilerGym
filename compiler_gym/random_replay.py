# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Replay the sequence of actions that produced the best reward."""
from pathlib import Path
from typing import List

from deprecated import deprecated

from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.random_search import replay_actions as replay_actions_
from compiler_gym.random_search import (
    replay_actions_from_logs as replay_actions_from_logs_,
)


@deprecated(version="0.2.1", reason="Use env.step(actions) instead")
def replay_actions(env: CompilerEnv, action_names: List[str], outdir: Path):
    return replay_actions_(env, action_names, outdir)


@deprecated(
    version="0.2.1",
    reason="Use compiler_gym.random_search.replay_actions_from_logs() instead",
)
def replay_actions_from_logs(env: CompilerEnv, logdir: Path, benchmark=None) -> None:
    return replay_actions_from_logs_(env, logdir, benchmark)
