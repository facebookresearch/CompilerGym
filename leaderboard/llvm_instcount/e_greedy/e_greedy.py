# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""ϵ-greedy policy for LLVM codesize."""
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from absl import flags

from compiler_gym.envs import CompilerEnv, LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

flags.DEFINE_float(
    "epsilon", 0, "The ratio of patience to the size of the action space. "
)
FLAGS = flags.FLAGS


class RewardAction(NamedTuple):
    """An action -> reward tuple for a single step()."""

    # Use reward as the element in the tuple as the reward is used for ordering.
    reward: float
    action: int


def select_best_action(env: CompilerEnv, executor: ThreadPoolExecutor) -> RewardAction:
    """Determine the best action by trying all possible options and ranking them."""

    def eval_action(fkd: CompilerEnv, action: int) -> RewardAction:
        """Evaluate the given action."""
        try:
            _, reward, _, _ = fkd.step(action)
        finally:
            fkd.close()
        return RewardAction(reward=reward, action=action)

    # Select the best action using the reward that the action produces, then
    # action index as a tie-breaker. Do this by creating n forks of the
    # environment, one for every action, and evaluting the actions in parallel
    # threads. Note that calls to fork() occur in the main thread for thread
    # safety in case of environment restart.
    futures = (
        executor.submit(eval_action, env.fork(), action)
        for action in range(env.action_space.n)
    )
    best_reward_action = RewardAction(reward=-float("inf"), action=0)
    for future in as_completed(futures):
        reward_action: RewardAction = future.result()
        if reward_action > best_reward_action:
            best_reward_action = reward_action

    return best_reward_action


def e_greedy_search(env: LlvmEnv) -> None:
    """Run an ϵ-greedy search on an environment."""
    step_count = 0
    with ThreadPoolExecutor(max_workers=FLAGS.nproc) as executor:
        while True:
            step_count += 1

            if random.random() < FLAGS.epsilon:
                # Exploratory step. Randomly select and apply an action.
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
                logging.debug(
                    "Step %d, exploratory action %s, reward %.4f, cumulative %.4f",
                    step_count,
                    env.action_space.flags[action],
                    reward,
                    env.episode_reward,
                )
            else:
                # Select the best reward and apply it, or terminate the search
                # if no positive reward is attainable.
                best = select_best_action(env, executor)
                if best.reward <= 0:
                    logging.debug(
                        "Greedy search terminated after %d steps, "
                        "no further reward attainable",
                        step_count,
                    )
                    done = True
                else:
                    _, reward, done, _ = env.step(best.action)
                    logging.debug(
                        "Step %d, greedy action %s, reward %.4f, cumulative %.4f",
                        step_count,
                        env.action_space.flags[best.action],
                        reward,
                        env.episode_reward,
                    )
                    if env.reward_space.deterministic and reward != best.reward:
                        logging.warning(
                            "Action %s produced different reward on replay, %.4f != %.4f",
                            env.action_space.flags[best.action],
                            best.reward,
                            reward,
                        )

                # Stop the search if we have reached a terminal state.
                if done:
                    return


if __name__ == "__main__":
    eval_llvm_instcount_policy(e_greedy_search)
