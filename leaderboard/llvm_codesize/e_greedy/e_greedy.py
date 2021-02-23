# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""ϵ-greedy policy for LLVM codesize."""
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from absl import flags

from compiler_gym.envs import CompilerEnv, LlvmEnv

# Import the ../eval_policy.py helper.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/..")
from eval_policy import eval_policy  # noqa

flags.DEFINE_float(
    "epsilon", 0, "The ratio of patience to the size of the action space. "
)
FLAGS = flags.FLAGS


class ActionResult(NamedTuple):
    """The step() result for an environment."""

    action: int
    reward: float


def eval_action(env: CompilerEnv, action: int) -> ActionResult:
    """Evaluate the given action."""
    fkd = env.fork()
    try:
        _, reward, _, _ = fkd.step(action)
    finally:
        fkd.close()
    return ActionResult(action=action, reward=reward)


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
                # Create an ordered list of actions, best first, ordered by
                # descding reward, using action index as a tie-breaker..
                futures = [
                    executor.submit(eval_action, env, action)
                    for action in range(env.action_space.n)
                ]
                results = [future.result() for future in as_completed(futures)]
                results = sorted(
                    results,
                    key=lambda result: (result.reward, result.action),
                    reverse=True,
                )

                # Select the best reward and apply it, or terminate the search
                # if no positive reward is attainable.
                if results[0].reward <= 0:
                    logging.debug(
                        "Greedy search terminated after %d steps, "
                        "no further reward attainable",
                        step_count,
                    )
                    return
                else:
                    action, expected_reward = results[0].action, results[0].reward
                    _, reward, done, _ = env.step(action)
                    logging.debug(
                        "Step %d, greedy action %s, reward %.4f, cumulative %.4f",
                        step_count,
                        env.action_space.flags[action],
                        reward,
                        env.episode_reward,
                    )
                    if env.reward_space.deterministic and reward != expected_reward:
                        logging.warning(
                            "Action %s produced different reward when reapplying, %.4f != %.4f",
                            env.action_space.flags[action],
                            reward,
                            expected_reward,
                        )

                # Stop the search if we have reached a terminal state.
                if done:
                    return


if __name__ == "__main__":
    eval_policy(e_greedy_search)
