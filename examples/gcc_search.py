# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from itertools import islice
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from absl import app, flags

import compiler_gym
import compiler_gym.util.flags.seed  # noqa Flag definition.
from compiler_gym.envs.gcc import DEFAULT_GCC, GccEnv
from compiler_gym.service import ServiceError
from compiler_gym.util.runfiles_path import create_user_logs_dir

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "gcc_bin", DEFAULT_GCC, "Binary to use for gcc. Use docker:<image> for docker"
)
flags.DEFINE_list(
    "gcc_benchmark", None, "List of benchmarks to search. Use 'all' for all"
)
flags.DEFINE_enum(
    "search",
    "random",
    ["random", "action-walk", "action-climb", "genetic"],
    "Type of search to perform",
)
flags.DEFINE_integer(
    "timeout", 20, "Timeout for each compilation in seconds", lower_bound=1
)
flags.DEFINE_string(
    "log",
    None,
    "Filename to log progress. "
    "Defaults to ~/logs/compiler_gym/gcc_autotuning/<timestamp>/log.csv",
)
flags.DEFINE_integer(
    "n", 100, "Maximum number of compilations per benchmark", lower_bound=1
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
        self.n = FLAGS.n

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

        scale = self.baseline.size / pt.size if pt.size != 0 else "-"
        with open(self.logfile, "a") as f:
            print(f"{scale}, {pt.size}, {n}, {','.join(map(str, pt.choices))}", file=f)

        print(
            f"{bname} scale={scale}, size={pt.size}, n={n}, choices={','.join(map(lambda c: str(c) if c != -1 else '-', pt.choices))}"
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
        for i in range(FLAGS.actions_per_step):
            env.step(env.action_space.sample())
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
        for i in range(FLAGS.actions_per_step):
            env.step(env.action_space.sample())
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


def main(argv):
    del argv  # Unused.

    search_map = {
        "random": RandomChoicesSearch,
        "action-walk": RandomWalkActionsSearch,
        "action-climb": HillClimbActionsSearch,
        "genetic": GAChoicesSearch,
    }
    search_cls = search_map[FLAGS.search]

    if FLAGS.seed:
        random.seed(FLAGS.seed)

    def get_benchmarks():
        benchmarks = []
        env = compiler_gym.make("gcc-v0", gcc_bin=FLAGS.gcc_bin)
        env.reset()
        for dataset in env.datasets:
            benchmarks += islice(dataset.benchmark_uris(), 50)
        env.close()
        benchmarks.sort()
        return benchmarks

    if not FLAGS.gcc_benchmark:
        print("Benchmark not given")
        print("Select from:")
        print("\n".join(get_benchmarks()))
        return

    if FLAGS.gcc_benchmark == ["all"]:
        benchmarks = get_benchmarks()
    else:
        benchmarks = FLAGS.gcc_benchmark

    logfile = (
        FLAGS.log
        if FLAGS.log
        else (create_user_logs_dir("gcc_autotuning") / "logs.csv")
    )
    print("Logging results to", logfile)

    searches = [
        search_cls(logfile=logfile, benchmark=benchmark) for benchmark in benchmarks
    ]
    for search in searches:
        search.run()

    for search in searches:
        print(search.benchmark, search.best)

    for search in searches:
        print(
            search.benchmark,
            search.best.size,
            search.baseline.size,
            search.baseline.size / search.best.size,
        )


if __name__ == "__main__":
    app.run(main)
