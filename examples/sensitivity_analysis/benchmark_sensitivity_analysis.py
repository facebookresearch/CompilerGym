# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Determine the typical reward delta of a benchmark using random trials.

This script estimates the change in reward that running a random episode has
on a benchmark by running trials. A trial is an episode in which a random
number of random actions are performed. Reward delta is the amount that the
reward signal changes from the initial to final action:
(reward_end - reward_init) / reward_init.

Example Usage
-------------

Evaluate the impact on LLVM codesize of random actions on the cBench-crc32
benchmark:

    $ bazel run -c opt //compiler_gym/bin:benchmark_sensitivity_analysis -- \
        --env=llvm-v0 --reward=IrInstructionCountO3 \
        --benchmark=cBench-crc32 --num_trials=50

Evaluate the LLVM codesize reward delta on all benchmarks:

    $ bazel run -c opt //compiler_gym/bin:benchmark_sensitivity_analysis -- \
        --env=llvm-v0 --reward=IrInstructionCountO3
"""
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.service.proto import Benchmark
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
    "The number of trials to perform when estimating the reward delta of each benchmark. "
    "A trial is a random episode of a benchmark. Increasing this number increases the "
    "number of trials performed, leading to a higher fidelity estimate of the reward "
    "potential for a benchmark.",
)
flags.DEFINE_integer(
    "min_steps",
    10,
    "The minimum number of random steps to make in a single trial.",
)
flags.DEFINE_integer(
    "max_steps",
    100,
    "The maximum number of random steps to make in a single trial.",
)
flags.DEFINE_integer(
    "nproc", cpu_count(), "The number of parallel evaluation threads to run."
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
    benchmark: Union[Benchmark, str],
    reward_space: str,
    num_trials: int,
    min_steps: int,
    max_steps: int,
    max_attempts_multiplier: int = 5,
) -> SensitivityAnalysisResult:
    """Run random trials to get a list of num_trials reward deltas."""
    rewards, runtimes = [], []
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
            benchmark = env.benchmark
            with Timer() as t:
                reward = run_one_trial(env, reward_space, min_steps, max_steps)
            if reward is not None:
                rewards.append(reward)
                runtimes.append(t.time)

    return SensitivityAnalysisResult(
        name=env.benchmark, runtimes=np.array(runtimes), rewards=np.array(rewards)
    )


def run_one_trial(
    env: CompilerEnv, reward_space: str, min_steps: int, max_steps: int
) -> Optional[float]:
    """Run a random number of "warmup" steps in an environment, then compute
    the reward delta of the given action.

        :return: The ratio of reward improvement.
    """
    num_steps = random.randint(min_steps, max_steps)
    init_reward = env.reward[reward_space]
    assert init_reward is not None
    for _ in range(num_steps):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            return None
    reward_after = env.reward[reward_space]
    assert reward_after is not None
    return reward_after


def run_benchmark_sensitivity_analysis(
    benchmarks: List[Union[Benchmark, str]],
    rewards_path: Path,
    runtimes_path: Path,
    reward: str,
    num_trials: int,
    min_steps: int,
    max_steps: int,
    nproc: int = cpu_count(),
    max_attempts_multiplier: int = 5,
):
    """Estimate the reward delta of a given list of benchmarks."""
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        analysis_futures = [
            executor.submit(
                get_rewards,
                benchmark,
                reward,
                num_trials,
                min_steps,
                max_steps,
                max_attempts_multiplier,
            )
            for benchmark in benchmarks
        ]
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

    # Determine the benchmark that is being analyzed, or use all of them.
    benchmark = benchmark_from_flags()
    if benchmark:
        benchmarks = [benchmark]
    else:
        with env_session_from_flags() as env:
            benchmarks = islice(env.benchmarks, 100)

    logs_dir = Path(
        FLAGS.output_dir or create_logging_dir("benchmark_sensitivity_analysis")
    )
    rewards_path = logs_dir / f"benchmarks_{FLAGS.reward}.csv"
    runtimes_path = logs_dir / f"benchmarks_{FLAGS.reward}_runtimes.csv"

    run_benchmark_sensitivity_analysis(
        rewards_path=rewards_path,
        runtimes_path=runtimes_path,
        benchmarks=benchmarks,
        reward=FLAGS.reward,
        num_trials=FLAGS.num_trials,
        min_steps=FLAGS.min_steps,
        max_steps=FLAGS.max_steps,
        nproc=FLAGS.nproc,
        max_attempts_multiplier=FLAGS.max_attempts_multiplier,
    )


if __name__ == "__main__":
    app.run(main)
