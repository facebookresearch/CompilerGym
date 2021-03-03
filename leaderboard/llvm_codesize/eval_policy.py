# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a helper function for evaluating LLVM codesize reduction
policies.

Usage:

    from compiler_gym.envs import LlvmEnv
    from eval_policy import eval_policy

    class MyLlvmCodesizePolicy:
        def __call__(env: LlvmEnv) -> None:
            pass # ...

    if __name__ == "__main__":
        eval_policy(MyLlvmCodesizePolicy())
"""
import platform
import sys
from collections import Counter
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Callable, Iterable

import GPUtil
import gym
import humanize
import psutil
from absl import app, flags
from cpuinfo import get_cpu_info
from tqdm import tqdm

import compiler_gym  # noqa Register environments.
from compiler_gym import CompilerEnvState
from compiler_gym.bin.validate import main as validate
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.timer import Timer

flags.DEFINE_string(
    "logfile", "results.csv", "The path of the file to write results to."
)
flags.DEFINE_string(
    "hardware_info",
    "hardware.txt",
    "The path of the file to write a hardware summary to.",
)
flags.DEFINE_integer(
    "max_benchmarks",
    0,
    "If > 0, use only the the first --max_benchmarks benchmarks from the "
    "dataset, as determined by alphabetical sort. If not set, all benchmarks "
    "from the dataset are used.",
)
flags.DEFINE_integer(
    "n", 10, "The number of repetitions of the search to run for each benchmark."
)
flags.DEFINE_string("test_dataset", "cBench-v0", "The dataset to use for the search.")
flags.DEFINE_boolean("validate", True, "Run validation on the results.")
flags.DEFINE_boolean(
    "resume",
    False,
    "If true, read the --logfile first and run only the policy evaluations not "
    "already in the logfile.",
)
FLAGS = flags.FLAGS

# A policy is a function that accepts as input an LLVM environment, and
# interacts with that environment with the goal of maximising cumulative reward.
Policy = Callable[[LlvmEnv], None]


def _get_gpus() -> Iterable[str]:
    """Return GPU info strings."""
    gpus = GPUtil.getGPUs()
    if gpus:
        yield from (gpu.name for gpu in gpus)
    else:
        yield "None"


def _get_cpu() -> str:
    """Return CPU info string."""
    cpuinfo = get_cpu_info()
    brand = cpuinfo["brand_raw"].replace("(R)", "")
    return f"{brand} ({cpuinfo['count']}× core)"


def _get_memory() -> str:
    """Return system memory info string."""
    return humanize.naturalsize(psutil.virtual_memory().total, binary=True)


def _get_os() -> str:
    """Return operating system name as a string."""
    return platform.platform()


def _summarize_duplicates(iterable: Iterable[str]) -> Iterable[str]:
    """Aggregate duplicates in a list of strings."""
    freq = sorted(Counter(iterable).items(), key=lambda x: -x[1])
    for gpu, count in freq:
        if count > 1:
            yield f"{count}x {gpu}"
        else:
            yield gpu


def _print_hardarwe_info(logfile):
    """Print a summary of system hardware to file."""
    print(
        tabulate(
            [
                ("OS", _get_os()),
                ("CPU", _get_cpu()),
                ("Memory", _get_memory()),
                ("GPU", ", ".join(_summarize_duplicates(_get_gpus()))),
            ],
            headers=("", "Hardware Specification"),
        ),
        file=logfile,
    )


class _BenchmarkRunner(Thread):
    def __init__(self, env, benchmarks, policy, print_header):
        super().__init__()
        self.env = env
        self.benchmarks = benchmarks
        self.policy = policy
        self.print_header = print_header
        self.n = 0

    def run(self):
        with open(FLAGS.logfile, "a") as logfile:
            for benchmark in self.benchmarks:
                self.n += 1
                self.env.reset(benchmark=benchmark)
                with Timer() as timer:
                    self.policy(self.env)

                # Sanity check that the policy didn't change the expected
                # experimental setup.
                assert self.env.in_episode, "Environment is no longer in an episode"
                assert (
                    self.env.benchmark == benchmark
                ), "Policy changed environment benchmark"
                assert self.env.reward_space, "Policy unset environment reward space"
                assert (
                    self.env.reward_space.id == "IrInstructionCountOz"
                ), "Policy changed environment reward space"

                # Override walltime in the generated state.
                state = CompilerEnvState(
                    benchmark=self.env.state.benchmark,
                    reward=self.env.state.reward,
                    walltime=timer.time,
                    commandline=self.env.state.commandline,
                )
                if self.print_header:
                    print(self.env.state.csv_header(), file=logfile)
                    self.print_header = False
                print(state.to_csv(), file=logfile, flush=True)


def eval_policy(policy: Policy) -> None:
    """Evaluate a policy on a target dataset.

    A policy is a function that takes as input an LlvmEnv environment and
    performs a set of actions on it.
    """

    def main(argv):
        assert len(argv) == 1, f"Unknown args: {argv[:1]}"
        assert FLAGS.n > 0, "n must be > 0"

        print(
            f"Writing inference results to '{FLAGS.logfile}' and "
            f"hardware summary to '{FLAGS.hardware_info}'"
        )

        with open(FLAGS.hardware_info, "w") as f:
            _print_hardarwe_info(f)

        env = gym.make("llvm-ic-v0")
        try:
            # Install the required dataset and build the list of benchmarks to
            # evaluate.
            env.require_dataset(FLAGS.test_dataset)
            benchmarks = sorted([b for b in env.benchmarks if FLAGS.test_dataset in b])
            if FLAGS.max_benchmarks:
                benchmarks = benchmarks[: FLAGS.max_benchmarks]

            # Repeat the searches for the requested number of iterations.
            benchmarks *= FLAGS.n
            benchmarks = sorted(benchmarks)
            total = len(benchmarks)

            # If we are resuming from a previous job, read the states that have
            # already been proccessed and remove those benchmarks from the list
            # of benchmarks to evaluate.
            print_header = True
            init = 0
            if Path(FLAGS.logfile).is_file():
                if FLAGS.resume:
                    with open(FLAGS.logfile, "r") as f:
                        for state in CompilerEnvState.read_csv_file(f):
                            init += 1
                            benchmarks.remove(state.benchmark)
                            print_header = False
                else:
                    Path(FLAGS.logfile).unlink()

            # Run the benchmark loop in background so that we can asynchronously
            # log progress.
            pbar = tqdm(initial=init, total=total, file=sys.stderr, unit=" benchmark")
            benchmark_runner = _BenchmarkRunner(env, benchmarks, policy, print_header)
            benchmark_runner.start()
            while benchmark_runner.is_alive():
                pbar.n = benchmark_runner.n + init
                pbar.refresh()
                sleep(1)
        finally:
            env.close()

        if FLAGS.validate:
            FLAGS.env = "llvm-ic-v0"
            validate(["argv0", FLAGS.logfile])

    app.run(main)
