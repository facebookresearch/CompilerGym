# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LLVM is a popular open source compiler used widely in industry and research.
The :code:`llvm-ic-v0` environment exposes LLVM's optimizing passes as a set of
actions that can be applied to a particular program. The goal of the agent is to
select the sequence of optimizations that lead to the greatest reduction in
instruction count in the program being compiled. Reward is the reduction in
instruction count achieved scaled to the reduction achieved by LLVM's builtin
:code:`-Oz` pipeline.

+--------------------+------------------------------------------------------+
| Property           | Value                                                |
+====================+======================================================+
| Environment        | :class:`LlvmEnv <compiler_gym.envs.LlvmEnv>`.        |
+--------------------+------------------------------------------------------+
| Observation Space  | Any.                                                 |
+--------------------+------------------------------------------------------+
| Reward Space       | Instruction count reduction relative to :code:`-Oz`. |
+--------------------+------------------------------------------------------+
| Test Dataset       | The 23 cBench benchmarks.                            |
+--------------------+------------------------------------------------------+

Users who wish to create a submission for this leaderboard may use
:func:`eval_llvm_instcount_policy()
<compiler_gym.leaderboard.llvm_instcount.eval_llvm_instcount_policy>` to
automatically evaluate their agent on the test set.
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

import compiler_gym.envs  # noqa Register environments.
from compiler_gym.bin.validate import main as validate
from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.timer import Timer

flags.DEFINE_string(
    "results_logfile", "results.csv", "The path of the file to write results to."
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
flags.DEFINE_string("test_dataset", "cBench-v1", "The dataset to use for the search.")
flags.DEFINE_boolean("validate", True, "Run validation on the results.")
flags.DEFINE_boolean(
    "resume",
    False,
    "If true, read the --results_logfile first and run only the policy "
    "evaluations not already in the logfile.",
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
    return f"{brand} ({cpuinfo['count']}x core)"


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


def _get_hardarwe_info() -> str:
    """Print a summary of system hardware to file."""
    return tabulate(
        [
            ("OS", _get_os()),
            ("CPU", _get_cpu()),
            ("Memory", _get_memory()),
            ("GPU", ", ".join(_summarize_duplicates(_get_gpus()))),
        ],
        headers=("", "Hardware Specification"),
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
        with open(FLAGS.results_logfile, "a") as logfile:
            for benchmark in self.benchmarks:
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
                self.n += 1


def eval_llvm_instcount_policy(policy: Policy) -> None:
    """Evaluate an LLVM codesize policy and generate results for a leaderboard
    submission.

    To use it, you define your policy as a function that takes an
    :class:`LlvmEnv <compiler_gym.envs.LlvmEnv>` instance as input and modifies
    it in place. For example, for a trivial random policy:

        >>> from compiler_gym.envs import LlvmEnv
        >>> def my_policy(env: LlvmEnv) -> None:
        ....   # Defines a policy that takes 10 random steps.
        ...    for _ in range(10):
        ...        _, _, done, _ = env.step(env.action_space.sample())
        ...        if done: break

    If your policy is stateful, you can use a class and override the
    :code:`__call__()` method:

        >>> class MyPolicy:
        ...     def __init__(self):
        ...         self.my_stateful_vars = {}  # or similar
        ...     def __call__(self, env: LlvmEnv) -> None:
        ...         pass # ... do fun stuff!
        >>> my_policy = MyPolicy()

    The role of your policy is to perform a sequence of actions on the supplied
    environment so as to maximize cumulative reward. By default, no observation
    space is set on the environment, so :meth:`env.step()
    <compiler_gym.envs.CompilerEnv.step>` will return :code:`None` for the
    observation. You may set a new observation space:

        >>> env.observation_space = "InstCount"  # Set a new space for env.step()
        >>> env.observation["InstCount"]  # Calculate a one-off observation.

    However, the policy may not change the reward space of the environment, or
    the benchmark.

    Once you have defined your policy, call the
    :func:`eval_llvm_instcount_policy()
    <compiler_gym.leaderboard.llvm_instcount.eval_llvm_instcount_policy>` helper
    function, passing it your policy as its only argument:

    >>> eval_llvm_instcount_policy(my_policy)

    Put together as a complete example, a leaderboard submission script may look
    like:

    .. code-block:: python

        # my_policy.py
        from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
        from compiler_gym.envs import LlvmEnv

        def my_policy(env: LlvmEnv) -> None:
            env.observation_space = "InstCount"  # we're going to use instcount space
            pass # ... do fun stuff!

        if __name__ == "__main__":
            eval_llvm_instcount_policy(my_policy)

    The :func:`eval_llvm_instcount_policy()
    <compiler_gym.leaderboard.llvm_instcount.eval_llvm_instcount_policy>` helper
    defines a number of commandline flags that can be overriden to control the
    behavior of the evaluation. For example the flag :code:`--n` determines the
    number of times the policy is run on each benchmark (default is 10), and
    :code:`--results_logfile` determines the path of the generated results file:

    .. code-block::

        $ python my_policy.py --n=5 --results_logfile=my_policy_results.csv

    You can use :code:`--helpfull` flag to list all of the flags that are
    defined:

    .. code-block::

        $ python my_policy.py --helpfull

    Once you are happy with your approach, see the `contributing guide
    <https://github.com/facebookresearch/CompilerGym/blob/development/CONTRIBUTING.md#leaderboard-submissions>`_
    for instructions on preparing a submission to the leaderboard.
    """

    def main(argv):
        assert len(argv) == 1, f"Unknown args: {argv[:1]}"
        assert FLAGS.n > 0, "n must be > 0"

        print(
            f"Writing inference results to '{FLAGS.results_logfile}' and "
            f"hardware summary to '{FLAGS.hardware_info}'"
        )

        with open(FLAGS.hardware_info, "w") as f:
            print(_get_hardarwe_info(), file=f)

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
            if Path(FLAGS.results_logfile).is_file():
                if FLAGS.resume:
                    with open(FLAGS.results_logfile, "r") as f:
                        for state in CompilerEnvState.read_csv_file(f):
                            if state.benchmark in benchmarks:
                                init += 1
                                benchmarks.remove(state.benchmark)
                                print_header = False
                else:
                    Path(FLAGS.results_logfile).unlink()

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
            validate(["argv0", FLAGS.results_logfile])

    app.run(main)
