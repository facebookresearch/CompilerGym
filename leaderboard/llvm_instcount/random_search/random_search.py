# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An implementation of a random search policy for the LLVM codesize task.

The search is the same as the included compiler_gym.bin.random_search. See
README.md in this directory for a detailed description.
"""
from time import sleep

import gym
from absl import flags

from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.random_search import RandomAgentWorker

flags.DEFINE_float(
    "patience_ratio",
    1.0,
    "The ratio of patience to the size of the action space. "
    "Patience = patience_ratio * action_space_size",
)
flags.DEFINE_integer(
    "search_time",
    60,
    "The minimum number of seconds to run the random search for. After this "
    "many seconds have elapsed the best results are aggregated from the "
    "search threads and the search is terminated.",
)
FLAGS = flags.FLAGS


def random_search(env: LlvmEnv) -> None:
    """Run a random search on the given environment."""
    patience = int(env.action_space.n * FLAGS.patience_ratio)

    # Start parallel random search workers.
    workers = [
        RandomAgentWorker(
            make_env=lambda: gym.make("llvm-ic-v0", benchmark=env.benchmark),
            patience=patience,
        )
        for _ in range(FLAGS.nproc)
    ]
    for worker in workers:
        worker.start()

    sleep(FLAGS.search_time)

    # Stop the workers.
    for worker in workers:
        worker.alive = False
    for worker in workers:
        worker.join()

    # Aggregate the best results.
    best_actions = []
    best_reward = -float("inf")
    for worker in workers:
        if worker.best_returns > best_reward:
            best_reward, best_actions = worker.best_returns, list(worker.best_actions)

    # Replay the best sequence of actions to produce the final environment
    # state.
    for action in best_actions:
        _, _, done, _ = env.step(action)
        assert not done


if __name__ == "__main__":
    eval_llvm_instcount_policy(random_search)
