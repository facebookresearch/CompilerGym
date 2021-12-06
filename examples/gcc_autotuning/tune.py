# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Autotuning script for GCC command line options."""
import random
from itertools import islice, product
from multiprocessing import Lock
from pathlib import Path
from typing import NamedTuple

import numpy as np
from absl import app, flags
from geneticalgorithm import geneticalgorithm as ga

import compiler_gym
import compiler_gym.util.flags.nproc  # noqa Flag definition.
import compiler_gym.util.flags.output_dir  # noqa Flag definition.
import compiler_gym.util.flags.seed  # noqa Flag definition.
from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.gcc import DEFAULT_GCC
from compiler_gym.service import ServiceError
from compiler_gym.util.executor import Executor
from compiler_gym.util.runfiles_path import create_user_logs_dir

from .info import info

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "gcc_bin", DEFAULT_GCC, "Binary to use for gcc. Use docker:<image> for docker"
)
flags.DEFINE_list(
    "gcc_benchmark",
    None,
    "List of benchmarks to search. Use 'all' for all. "
    "Defaults to the 12 CHStone benchmarks.",
)
flags.DEFINE_list(
    "search",
    ["random", "hillclimb", "genetic"],
    "Type of search to perform. One of: {random,hillclimb,genetic}",
)
flags.DEFINE_integer(
    "timeout", 60, "Timeout for each compilation in seconds", lower_bound=1
)
flags.DEFINE_integer(
    "gcc_search_budget",
    100,
    "Maximum number of compilations per benchmark",
    lower_bound=1,
)
flags.DEFINE_integer(
    "gcc_search_repetitions", 1, "Number of times to repeat each search", lower_bound=1
)
flags.DEFINE_integer(
    "actions_per_step",
    10,
    "Number of actions per compilation for action based searches",
    lower_bound=1,
)
flags.DEFINE_integer("max_range", 256, "Limit space per option", lower_bound=0)
flags.DEFINE_integer("pop_size", 100, "Population size for GA", lower_bound=1)
flags.DEFINE_enum(
    "objective", "obj_size", ["asm_size", "obj_size"], "Which objective to use"
)

# Lock to prevent multiple processes all calling compiler_gym.make("gcc-v0")
# simultaneously as this can cause issues with the docker API.
GCC_ENV_CONSTRUCTOR_LOCK = Lock()


def random_search(env: CompilerEnv):
    best = float("inf")
    for _ in range(FLAGS.gcc_search_budget):
        env.reset()
        env.choices = [
            random.randint(-1, min(FLAGS.max_range, len(opt) - 1))
            for opt in env.gcc_spec.options
        ]
        best = min(objective(env), best)
    return best


def hill_climb(env: CompilerEnv):
    best = float("inf")
    for _ in range(FLAGS.gcc_search_budget):
        with env.fork() as fkd:
            fkd.choices = [
                random.randint(
                    max(-1, x - 5), min(len(env.gcc_spec.options[i]) - 1, x + 5)
                )
                for i, x in enumerate(env.choices)
            ]
            cost = objective(fkd)
            if cost < objective(env):
                best = cost
                env.choices = fkd.choices
    return best


def genetic_algorithm(env: CompilerEnv):
    def f(choices):
        env.reset()
        env.choices = choices = list(map(int, choices))
        s = objective(env)
        return s if s > 0 else float("inf")

    model = ga(
        function=f,
        dimension=len(env.gcc_spec.options),
        variable_type="int",
        variable_boundaries=np.array(
            [[-1, min(FLAGS.max_range, len(opt) - 1)] for opt in env.gcc_spec.options]
        ),
        function_timeout=FLAGS.timeout,
        algorithm_parameters={
            "population_size": FLAGS.pop_size,
            "max_num_iteration": max(1, int(FLAGS.gcc_search_budget / FLAGS.pop_size)),
            "mutation_probability": 0.1,
            "elit_ratio": 0.01,
            "crossover_probability": 0.5,
            "parents_portion": 0.3,
            "crossover_type": "uniform",
            "max_iteration_without_improv": None,
        },
    )
    model.run()
    return model.best_function


def objective(env) -> int:
    """Get the objective from an environment"""
    # Retry loop to defend against flaky environment.
    for _ in range(5):
        try:
            return env.observation[FLAGS.objective]
        except ServiceError as e:
            print(f"Objective function failed: {e}")
            env.reset()
    return env.observation[FLAGS.objective]


_SEARCH_FUNCTIONS = {
    "random": random_search,
    "hillclimb": hill_climb,
    "genetic": genetic_algorithm,
}


class SearchResult(NamedTuple):
    search: str
    benchmark: str
    best_size: int
    baseline_size: int

    @property
    def scaled_best(self) -> float:
        return self.baseline_size / self.best_size


def run_search(search: str, benchmark: str, seed: int) -> SearchResult:
    """Run a search and return the search class instance."""
    with GCC_ENV_CONSTRUCTOR_LOCK:
        env = compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin)

    try:
        random.seed(seed)
        np.random.seed(seed)

        env.reset(benchmark=benchmark)
        env.step(env.action_space["-Os"])
        baseline_size = objective(env)
        env.reset(benchmark=benchmark)
        best_size = _SEARCH_FUNCTIONS[search](env)
    finally:
        env.close()

    return SearchResult(
        search=search,
        benchmark=benchmark,
        best_size=best_size,
        baseline_size=baseline_size,
    )


def main(argv):
    del argv  # Unused.

    # Validate the --search values now.
    for search in FLAGS.search:
        if search not in _SEARCH_FUNCTIONS:
            raise app.UsageError(f"Invalid --search value: {search}")

    def get_benchmarks():
        benchmarks = []
        with compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin) as env:
            env.reset()
            if FLAGS.gcc_benchmark == ["all"]:
                for dataset in env.datasets:
                    benchmarks += islice(dataset.benchmark_uris(), 50)
            elif FLAGS.gcc_benchmark:
                for uri in FLAGS.gcc_benchmark:
                    benchmarks.append(env.datasets.benchmark(uri).uri)
            else:
                benchmarks = list(
                    env.datasets["benchmark://chstone-v0"].benchmark_uris()
                )
        benchmarks.sort()
        return benchmarks

    logdir = (
        Path(FLAGS.output_dir)
        if FLAGS.output_dir
        else create_user_logs_dir("gcc_autotuning")
    )
    logdir.mkdir(exist_ok=True, parents=True)
    with open(logdir / "results.csv", "w") as f:
        print(
            "search",
            "benchmark",
            "scaled_size",
            "size",
            "baseline_size",
            sep=",",
            file=f,
        )
    print("Logging results to", logdir)

    # Parallel execution environment. Use flag --nproc to control the number of
    # worker processes.
    executor = Executor(type="local", timeout_hours=12, cpus=FLAGS.nproc, block=True)
    with executor.get_executor(logs_dir=logdir) as session:
        jobs = []
        # Submit each search instance as a separate job.
        grid = product(
            range(FLAGS.gcc_search_repetitions), FLAGS.search, get_benchmarks()
        )
        for _, search, benchmark in grid:
            if not benchmark:
                raise app.UsageError("Empty benchmark name not allowed")

            jobs.append(
                session.submit(
                    run_search,
                    search=search,
                    benchmark=benchmark,
                    seed=FLAGS.seed + len(jobs),
                )
            )

        for job in jobs:
            result = job.result()
            print(result.benchmark, f"{result.scaled_best:.3f}x", sep="\t")
            with open(logdir / "results.csv", "a") as f:
                print(
                    result.search,
                    result.benchmark,
                    result.scaled_best,
                    result.best_size,
                    result.baseline_size,
                    sep=",",
                    file=f,
                )

    # Print results aggregates.
    info([logdir])


if __name__ == "__main__":
    app.run(main)
