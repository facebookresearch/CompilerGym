# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Determine the typical reward delta of different actions using random trials.

This script estimates the change in reward that running a specific action has
by running trials. A trial is a random episode that ends with the determined
action. Reward delta is the amount that the reward signal changes from running
that action: (reward_after - reward_before) / reward_before.

Example Usage
-------------

Evaluate the impact of three passes on the codesize of the cBench-crc32
benchmark:

    $ bazel run -c opt //compiler_gym/bin:action_sensitivity_analysis -- \
        --env=llvm-v0 --reward=IrInstructionCountO3 \
        --benchmark=cbench-v1/crc32 --num_trials=100 \
        --action=AddDiscriminatorsPass,AggressiveDcepass,AggressiveInstCombinerPass

Evaluate the single-step reward delta of all actions on LLVM codesize:

    $ bazel run -c opt //compiler_gym/bin:action_ensitivity_analysis -- \
        --env=llvm-v0 --reward=IrInstructionCountO3
"""
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional

import numpy as np
from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_session_from_flags
from compiler_gym.util.logs import create_logging_dir
from compiler_gym.util.timer import Timer
from examples.sensitivity_analysis.sensitivity_analysis_eval import (
    SensitivityAnalysisResult,
    run_sensitivity_analysis,
)

flags.DEFINE_integer(
    "num_trials",
    100,
    "The number of trials to perform when estimating the reward of each action. "
    "A trial is a random episode that ends with the determined action. Increasing "
    "this number increases the number of trials performed, leading to a higher "
    "fidelity estimate of the reward of an action.",
)
flags.DEFINE_integer(
    "max_warmup_steps",
    25,
    "The maximum number of random steps to make before determining the reward of an action.",
)
flags.DEFINE_integer(
    "nproc", cpu_count(), "The number of parallel evaluation threads to run."
)
flags.DEFINE_list(
    "action",
    [],
    "An optional list of actions to evaluate. If not specified, all actions will be evaluated.",
)
flags.DEFINE_integer(
    "max_attempts_multiplier",
    5,
    "A trial may fail because the environment crashes, or an action produces an invalid state. "
    "Limit the total number of trials performed for each action to "
    "max_attempts_multiplier * num_trials.",
)

FLAGS = flags.FLAGS


def get_rewards(
    action: int,
    action_name: str,
    reward_space: str,
    num_trials: int,
    max_warmup_steps: int,
    max_attempts_multiplier: int = 5,
) -> SensitivityAnalysisResult:
    """Run random trials to get a list of num_trials reward deltas."""
    rewards, runtimes = [], []
    benchmark = benchmark_from_flags()
    num_attempts = 0
    while (
        num_attempts < max_attempts_multiplier * num_trials
        and len(rewards) < num_trials
    ):
        num_attempts += 1
        with env_session_from_flags(benchmark=benchmark) as env:
            env.observation_space = None
            env.reward_space = None
            env.reset(benchmark=benchmark)
            with Timer() as t:
                reward = run_one_trial(env, reward_space, action, max_warmup_steps)
            if reward is not None:
                rewards.append(reward)
                runtimes.append(t.time)

    return SensitivityAnalysisResult(
        name=action_name, runtimes=np.array(runtimes), rewards=np.array(rewards)
    )


def run_one_trial(
    env: CompilerEnv, reward_space: str, action: int, max_warmup_steps: int
) -> Optional[float]:
    """Run a random number of "warmup" steps in an environment, then compute
    the reward delta of the given action.

        :return: The ratio of reward improvement.
    """
    num_warmup_steps = random.randint(0, max_warmup_steps)
    for _ in range(num_warmup_steps):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            return None
    # Force reward calculation.
    init_reward = env.reward[reward_space]
    assert init_reward is not None
    _, _, done, _ = env.step(action)
    if done:
        return None
    reward_after = env.reward[reward_space]
    assert reward_after is not None
    return reward_after


def run_action_sensitivity_analysis(
    actions: List[int],
    rewards_path: Path,
    runtimes_path: Path,
    reward_space: str,
    num_trials: int,
    max_warmup_steps: int,
    nproc: int = cpu_count(),
    max_attempts_multiplier: int = 5,
):
    """Estimate the reward delta of a given list of actions."""
    with env_session_from_flags() as env:
        action_names = env.action_space.names

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        analysis_futures = {
            executor.submit(
                get_rewards,
                action,
                action_names[action],
                reward_space,
                num_trials,
                max_warmup_steps,
                max_attempts_multiplier,
            )
            for action in actions
        }
        return run_sensitivity_analysis(
            analysis_futures=analysis_futures,
            runtimes_path=runtimes_path,
            rewards_path=rewards_path,
        )


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    with env_session_from_flags() as env:
        action_names = env.action_space.names

    if FLAGS.action:
        actions = [action_names.index(a) for a in FLAGS.action]
    else:
        actions = list(range(len(action_names)))

    logs_dir = Path(
        FLAGS.output_dir or create_logging_dir("benchmark_sensitivity_analysis")
    )
    rewards_path = logs_dir / f"actions_{FLAGS.reward}.rewards.csv"
    runtimes_path = logs_dir / f"actions_{FLAGS.reward}.runtimes.csv"

    run_action_sensitivity_analysis(
        rewards_path=rewards_path,
        runtimes_path=runtimes_path,
        actions=actions,
        reward=FLAGS.reward,
        num_trials=FLAGS.num_trials,
        max_warmup_steps=FLAGS.max_warmup_steps,
        nproc=FLAGS.nproc,
        max_attempts_multiplier=FLAGS.max_attempts_multiplier,
    )


if __name__ == "__main__":
    app.run(main)
