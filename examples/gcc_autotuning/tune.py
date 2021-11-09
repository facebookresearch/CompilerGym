# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Autotuning script for GCC command line options.
"""
import random
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from absl import app, flags
from gcc_autotuning.info import info

import compiler_gym
import compiler_gym.util.flags.nproc  # noqa Flag definition.
import compiler_gym.util.flags.output_dir  # noqa Flag definition.
import compiler_gym.util.flags.seed  # noqa Flag definition.
from compiler_gym.envs.gcc import DEFAULT_GCC, GccEnv
from compiler_gym.service import ServiceError
from compiler_gym.util.executor import Executor
from compiler_gym.util.runfiles_path import create_user_logs_dir

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
    "Type of search to perform. One of: {random,action-walk,hillclimb,genetic}",
)
flags.DEFINE_integer(
    "timeout", 20, "Timeout for each compilation in seconds", lower_bound=1
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


class ChoicesSearchPoint:
    """A pair of compilation choices and the resulting object file size."""

    def __init__(self, choices: List[int], size: Optional[int]):
        """If size is None then float.inf will be used to make comparisons easier."""
        self.choices = choices
        self.size = size if size is not None and size != -1 else float("inf")

    def better_than(self, other: "ChoicesSearchPoint") -> bool:
        """Determine if this result is better than the best so far.
        The choices are the list of choices to make.
        The size is the size of object file.
        Smaller size is better.
        If the sizes are the same, then the sums of the choices are used.
        """
        if self.size == other.size:
            if self.choices is not None and other.choices is not None:
                return sum(self.choices) < sum(other.choices)
            else:
                return self.choices
        return self.size < other.size

    def __str__(self) -> str:
        return f"{self.size} {self.choices}"


class ChoicesSearch:
    """Base class for searches."""

    def __init__(self, logfile: Path, benchmark: str):
        self.logfile = logfile
        self.logfile.touch()
        self.benchmark = benchmark
        # We record the best point as we go
        self.best = ChoicesSearchPoint(None, None)

        # Create an environment and get the baselin
        env = self.make()
        env.reset(benchmark=self.benchmark)
        self.gcc_spec = env.gcc_spec
        env.timeout = FLAGS.timeout
        env.step(env.action_space.names.index("-Os"))
        self.baseline = ChoicesSearchPoint(env.choices, self.objective(env))
        env.close()

        # The number of points to search
        self.n = FLAGS.gcc_search_budget

    def make(self) -> GccEnv:
        """Make an environment"""
        env = compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin)
        env.reset(benchmark=self.benchmark)
        env.timeout = FLAGS.timeout
        return env

    def objective(self, env) -> int:
        """Get the objective from an environment"""
        # Retry loop to defend against flaky environment.
        for _ in range(3):
            try:
                return env.observation[FLAGS.objective]
            except ServiceError:
                env.reset()
        return env.observation[FLAGS.objective]

    def step(self, env) -> ChoicesSearchPoint:
        """Take one search step"""
        raise NotImplementedError()

    def run(self):
        """Run the search."""
        env = self.make()

        while self.n > 0:
            n = self.n
            self.n -= 1

            env.reset(benchmark=self.benchmark)
            pt = self.step(env)
            self.log_pt(n, pt)

            if pt.better_than(self.best):
                self.best = pt
        env.close()

    def log_pt(self, n: int, pt: ChoicesSearchPoint):
        """Log the current point"""
        bname = self.benchmark.replace("benchmark://", "")

        scaled_size = self.baseline.size / pt.size if pt.size != 0 else "-"
        with open(self.logfile, "a") as f:
            print(self.benchmark, scaled_size, pt.size, n, *pt.choices, sep=",", file=f)

        print(
            f"{bname} scaled_size={scaled_size}, size={pt.size}, n={n}, choices={','.join(map(lambda c: str(c) if c != -1 else '-', pt.choices))}"
        )


class RandomChoicesSearch(ChoicesSearch):
    """A simple random search"""

    def random_choices(self) -> List:
        """Get a random set of choices"""
        return [
            random.randint(-1, min(FLAGS.max_range, len(opt) - 1))
            for opt in self.gcc_spec.options
        ]

    def step(self, env):
        choices = self.random_choices()
        env.choices = choices
        size = self.objective(env)
        return ChoicesSearchPoint(choices, size)


class RandomWalkActionsSearch(ChoicesSearch):
    """Randomly select actions"""

    def step(self, env):
        before = env.choices
        env.step([env.action_space.sample() for _ in range(FLAGS.actions_per_step)])
        after = env.choices
        size = self.objective(env)
        pt = ChoicesSearchPoint(after, size)
        if size != -1:
            env.choices = before
        return pt


class HillClimbActionsSearch(ChoicesSearch):
    """Randomly select actions and accept if they make things better"""

    def step(self, env):
        best = self.best.choices if self.best.choices is not None else env.choices
        env.choices = best
        env.step([env.action_space.sample() for _ in range(FLAGS.actions_per_step)])
        after = env.choices
        size = self.objective(env)
        return ChoicesSearchPoint(after, size)


class GAChoicesSearch(ChoicesSearch):
    """A simple, continuous genetic algorithm search"""

    init_fn = Callable[[], List[int]]
    selector_fn = Callable[[List[ChoicesSearchPoint]], ChoicesSearchPoint]
    xover_fn = Callable[[List[int], List[int]], List[int]]
    mutator_fn = Callable[[List[int]], List[int]]
    replace_fn = Callable[[List[List[int]]], int]

    def __init__(self, logfile: Path, benchmark: str):
        super().__init__(logfile=logfile, benchmark=benchmark)
        self.pop = []

        # Operators
        # Each is a list of functions that provide some capability, paired with
        # a set of weights tha control how likely the operator will be to be
        # chosen.

        # Inits are operators to create new choices
        self.inits: List[GAChoicesSearch.init_fn] = [self.init()]
        self.init_weights = [1] * len(self.inits)

        # Selectors choose individuals from the population
        self.selectors: List[GAChoicesSearch.selector_fn] = [self.tournament(7)]
        self.selector_weights = [1] * len(self.selectors)

        # Crossover operators, take two parents and produce a new child
        self.xovers: List[GAChoicesSearch.xover_fn] = [
            self.xover_npoint(),
            self.xover_between(),
        ]
        self.xover_weights = [1] * len(self.xovers)

        # Mutators take a set of choices and mess with them.
        # Note that the identity mutator is much more likely than the others
        self.mutators: List[GAChoicesSearch.mutator_fn] = [
            self.mutate_empty(1),
            self.mutate_empty(10),
            self.mutate_rand_elements(1),
            self.mutate_rand_elements(10),
            self.mutate_bump(1, 1),
            self.mutate_bump(1, 5),
            self.mutate_bump(5, 1),
            self.mutate_bump(5, 5),
            self.identity(),
        ]
        self.mutator_weights = [5, 1, 5, 1, 5, 1, 5, 1, 30]

        # These functions choose which individual gets replaced
        self.replacers: List[GAChoicesSearch.replacer_fn] = [
            self.replace_tournament(7),
            self.replace_tournament(3),
            self.replace_worst(),
        ]
        self.replacer_weights = [10, 5, 1]

    def init(self) -> "GAChoicesSearch.init_fn":
        """Returns a function that creates a random set of choices"""

        def random_choices() -> List[int]:
            return [
                random.randint(-1, min(FLAGS.max_range, len(opt) - 1))
                for opt in self.gcc_spec.options
            ]

        return random_choices

    def tournament(self, k: int = 7) -> "GAChoicesSearch.selector_fn":
        """Returns a function which will performa tournament selection with a
        tournament of the given size, k."""

        def key(pt: ChoicesSearchPoint) -> Tuple[int, int]:
            return (pt.size, sum(pt.choices))

        def select(pop: List[ChoicesSearchPoint]) -> List[int]:
            cands = random.sample(pop, k)
            return min(cands, key=key)

        return select

    def xover_npoint(self) -> "GAChoicesSearch.xover_fn":
        """Returns a function which does crossover. Given two lists of
        choices, a new list of the same size is produced. Each member is
        equally likely to be from either of the two parents."""

        def single(x: int, y: int) -> int:
            return x if bool(random.getrandbits(1)) else y

        def change(a: List[int], b: List[int]) -> List[int]:
            return [single(x, y) for x, y in zip(a, b)]

        return change

    def xover_between(self) -> "GAChoicesSearch.xover_fn":
        """Returns a function which does crossover. Given two lists of
        choices, a new list of the same size is produced. Each member is
        randomly chosen from the range of the corresponding elements in the two
        parents."""

        def single(x: int, y: int) -> int:
            return random.randint(min(x, y), max(x, y))

        def change(a: List[int], b: List[int]) -> List[int]:
            return [single(x, y) for x, y in zip(a, b)]

        return change

    def mutate_rand_elements(self, k: int) -> "GAChoicesSearch.mutator_fn":
        """Returns a mutation function. It will replace k elements from the
        passed in choices, randomly selecting from the available range."""

        def change(a: List[int]) -> List[int]:
            b = a.copy()
            n = len(a)
            for _ in range(k):
                i = random.randrange(n)
                b[i] = random.randrange(
                    -1, min(FLAGS.max_range, len(self.gcc_spec.options[i]) - 1)
                )
            return b

        return change

    def mutate_empty(self, k: int) -> "GAChoicesSearch.mutator_fn":
        """Returns a mutation function. It will replace k elements from the
        passed in choices with the lowest available value (-1)."""

        def change(a: List[int]) -> List[int]:
            b = a.copy()
            n = len(a)
            for _ in range(k):
                i = random.randrange(n)
                b[i] = -1
            return b

        return change

    def mutate_bump(self, k: int, d: int) -> "GAChoicesSearch.mutator_fn":
        """Returns a mutation function. It will increment k elements from the
        passed in choices by a random amount from [-d, d]"""

        def change(a: List[int]) -> List[int]:
            b = a.copy()
            n = len(a)
            for _ in range(k):
                i = random.randrange(n)
                p = random.randint(-d, d)
                b[i] = min(-1, max(len(self.gcc_spec.options[i]) - 1, b[i] + p))
            return b

        return change

    def identity(self) -> "GAChoicesSearch.mutator_fn":
        """Identity mutator. Does not change the choices."""
        return lambda a: a.copy()

    def replace_worst(self) -> "GAChoicesSearch.replacer_fn":
        """Returns a function which, given a population, will return the index
        of the worst member"""

        def key(pt: ChoicesSearchPoint) -> Tuple[int, int]:
            return (pt.size, sum(pt.choices))

        def index(pop: List[List[int]]) -> int:
            return pop.index(max(pop, key=key))

        return index

    def replace_tournament(self, k: int = 3) -> "GAChoicesSearch.replacer_fn":
        """Returns a function which, given a population, will return the index
        of the worst member"""

        def key(i: int) -> Tuple[int, int]:
            pt = self.pop[i]
            return (pt.size, sum(pt.choices))

        def index(pop: List[List[int]]) -> int:
            cands = random.sample(range(len(pop)), k)
            return max(cands, key=key)

        return index

    def step(self, env):
        if len(self.pop) < FLAGS.pop_size:
            # We need to create a new individual
            init = random.choices(self.inits, weights=self.init_weights)[0]
            choices = env.choices = init()
            size = self.objective(env)
            pt = ChoicesSearchPoint(choices, size)
            self.pop.append(pt)
        else:
            # Select two parents
            sel_a, sel_b = random.choices(
                self.selectors, weights=self.selector_weights, k=2
            )
            a = sel_a(self.pop)
            b = sel_b(self.pop)
            # Cross over
            xover = random.choices(self.xovers, weights=self.xover_weights)[0]
            choices = xover(a.choices, b.choices)
            # Mutation - until we get something different from a and b
            different = False
            while not different:
                mutate = random.choices(self.mutators, weights=self.mutator_weights)[0]
                choices = mutate(choices)
                different = choices != a.choices and choices != b.choices
            # Evaluate
            env.choices = choices
            size = self.objective(env)
            pt = ChoicesSearchPoint(choices, size)
            # Replace
            replace = random.choices(self.replacers, weights=self.replacer_weights)[0]
            i = replace(self.pop)
            self.pop[i] = pt

        # Report on the step
        oks = [x.size for x in self.pop if x.size != float("inf")]
        chs = [sum(x.choices) for x in self.pop]
        size_str = f"size: min={min(oks, default='-')} max={max(oks, default='-')} avg={sum(oks) / len(oks) if oks else '-'}"
        choices_str = (
            f"choices: min={min(chs)} max={max(chs)} avg={sum(chs) / len(self.pop)}"
        )
        print(f"pop={len(self.pop)} ok={len(oks)} {size_str} {choices_str}")
        return pt


_SEARCH_CLASS_MAP: Dict[str, ChoicesSearch] = {
    "random": RandomChoicesSearch,
    "action-walk": RandomWalkActionsSearch,
    "hillclimb": HillClimbActionsSearch,
    "genetic": GAChoicesSearch,
}


class SearchResult(NamedTuple):
    search: str
    benchmark: str
    best_size: int
    baseline_size: int

    @property
    def scaled_best(self) -> float:
        return self.baseline_size / self.best_size


def run_search(search: str, logfile: Path, benchmark: str) -> SearchResult:
    """Run a search and return the search class instance."""
    search_class = _SEARCH_CLASS_MAP[search]
    job: ChoicesSearch = search_class(logfile=logfile, benchmark=benchmark)
    job.run()
    return SearchResult(
        search=search,
        benchmark=benchmark,
        best_size=job.best.size,
        baseline_size=job.baseline.size,
    )


def main(argv):
    del argv  # Unused.

    # Validate the --search values now.
    for search in FLAGS.search:
        if search not in _SEARCH_CLASS_MAP:
            raise app.UsageError(f"Invalid --search value: {search}")

    if FLAGS.seed:
        random.seed(FLAGS.seed)

    def get_benchmarks_from_all_datasets():
        """Enumerate first 50 benchmarks from each dataset."""
        benchmarks = []
        with compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin) as env:
            env.reset()
            for dataset in env.datasets:
                benchmarks += islice(dataset.benchmark_uris(), 50)
        benchmarks.sort()
        return benchmarks

    def get_chstone_benchmark_uris() -> List:
        with compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin) as env:
            return list(env.datasets["benchmark://chstone-v0"].benchmark_uris())

    if FLAGS.gcc_benchmark == ["all"]:
        benchmarks = get_benchmarks_from_all_datasets()
    elif FLAGS.gcc_benchmark:
        benchmarks = FLAGS.gcc_benchmark
    else:
        benchmarks = get_chstone_benchmark_uris()

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
        for _ in range(FLAGS.gcc_search_repetitions):
            for search in FLAGS.search:
                for benchmark in benchmarks:
                    jobs.append(
                        session.submit(
                            run_search,
                            search=search,
                            logfile=logdir / f"search-job-{len(jobs):04d}-log.csv",
                            benchmark=benchmark,
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
