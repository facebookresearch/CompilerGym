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
import logging
import os
from itertools import islice
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Callable, List

import gym
import humanize
from absl import app, flags

import compiler_gym.envs  # noqa Register environments.
from compiler_gym.bin.validate import main as validate
from compiler_gym.compiler_env_state import (
    CompilerEnvState,
    CompilerEnvStateReader,
    CompilerEnvStateWriter,
)
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer, humanize_duration_hms

flags.DEFINE_string(
    "leaderboard_results",
    "llvm_instcount-results.csv",
    "The path of the file to write results to.",
)
flags.DEFINE_string(
    "leaderboard_logfile",
    "llvm_instcount-results.log",
    "The path of a file to stream CompilerGym logs to.",
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
flags.DEFINE_string("test_dataset", "cbench-v1", "The dataset to use for the search.")
flags.DEFINE_boolean("validate", True, "Run validation on the results.")
flags.DEFINE_boolean(
    "resume",
    False,
    "If true, read the --leaderboard_results file first and run only the "
    "evaluations not already in the results file.",
)
FLAGS = flags.FLAGS

# A policy is a function that accepts as input an LLVM environment, and
# interacts with that environment with the goal of maximising cumulative reward.
Policy = Callable[[LlvmEnv], None]


class _EvalPolicyWorker(Thread):
    """Worker thread to evaluate a policy."""

    def __init__(
        self,
        env: LlvmEnv,
        benchmarks: List[str],
        policy: Policy,
        init_states: List[CompilerEnvState],
    ):
        super().__init__()
        self.env = env
        self.benchmarks = benchmarks
        self.policy = policy
        self.states: List[CompilerEnvState] = init_states
        self.alive = True

    def run(self):
        # Determine if we need to print a header.
        header = (
            not Path(FLAGS.leaderboard_results).is_file()
            or os.stat(FLAGS.leaderboard_results).st_size == 0
        )
        with CompilerEnvStateWriter(
            open(FLAGS.leaderboard_results, "a"), header=header
        ) as writer:
            for benchmark in self.benchmarks:
                self.env.reset(benchmark=benchmark)
                with Timer() as timer:
                    self.policy(self.env)

                # Sanity check that the policy didn't change the expected
                # experimental setup.
                assert self.env.in_episode, "Environment is no longer in an episode"
                assert self.env.benchmark and (
                    self.env.benchmark == benchmark
                ), "Policy changed environment benchmark"
                assert self.env.reward_space, "Policy unset environment reward space"
                assert (
                    self.env.reward_space.id == "IrInstructionCountOz"
                ), "Policy changed environment reward space"

                # Override walltime in the generated state.
                state = self.env.state.copy()
                state.walltime = timer.time

                writer.write_state(state, flush=True)
                self.states.append(state)

                if not self.alive:
                    return


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
    :code:`--leaderboard_results` determines the path of the generated results file:

    .. code-block::

        $ python my_policy.py --n=5 --leaderboard_results=my_policy_results.csv

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

        env = gym.make("llvm-ic-v0")

        # Stream verbose CompilerGym logs to file.
        logger = logging.getLogger("compiler_gym")
        logger.setLevel(logging.DEBUG)
        env.logger.setLevel(logging.DEBUG)
        log_handler = logging.FileHandler(FLAGS.leaderboard_logfile)
        logger.addHandler(log_handler)
        logger.propagate = False

        print(f"Writing results to {FLAGS.leaderboard_results}")
        print(f"Writing logs to {FLAGS.leaderboard_logfile}")

        try:
            # Build the list of benchmarks to evaluate.
            benchmarks = env.datasets[FLAGS.test_dataset].benchmark_uris()
            if FLAGS.max_benchmarks:
                benchmarks = islice(benchmarks, FLAGS.max_benchmarks)
            benchmarks = list(benchmarks)

            # Repeat the searches for the requested number of iterations.
            benchmarks *= FLAGS.n
            benchmarks = sorted(benchmarks)
            total_count = len(benchmarks)

            # If we are resuming from a previous job, read the states that have
            # already been proccessed and remove those benchmarks from the list
            # of benchmarks to evaluate.
            init_states = []
            if FLAGS.resume and Path(FLAGS.leaderboard_results).is_file():
                with CompilerEnvStateReader(open(FLAGS.leaderboard_results)) as reader:
                    for state in reader:
                        init_states.append(state)
                        if state.benchmark in benchmarks:
                            benchmarks.remove(state.benchmark)

            # Run the benchmark loop in background so that we can asynchronously
            # log progress.
            worker = _EvalPolicyWorker(env, benchmarks, policy, init_states)
            worker.start()
            timer = Timer().reset()
            try:
                print(
                    f"=== Evaluating policy on "
                    f"{humanize.intcomma(total_count)} "
                    f"{FLAGS.test_dataset} benchmarks ==="
                    "\n\n"  # Blank lines will be filled below
                )
                while worker.is_alive():
                    done_count = len(worker.states)
                    remaining_count = total_count - done_count
                    time = timer.time
                    gmean_reward = geometric_mean([s.reward for s in worker.states])
                    mean_walltime = (
                        arithmetic_mean([s.walltime for s in worker.states]) or time
                    )
                    print(
                        "\r\033[2A"
                        "\033[K"
                        f"Runtime: {humanize_duration_hms(time)}. "
                        f"Estimated completion: {humanize_duration_hms(time + mean_walltime * remaining_count)}. "
                        f"Completed: {humanize.intcomma(done_count)} / {humanize.intcomma(total_count)} "
                        f"({done_count / total_count:.1%})."
                        "\n\033[K"
                        f"Current mean walltime: {mean_walltime:.3f}s / benchmark."
                        "\n\033[K"
                        f"Current geomean reward: {gmean_reward:.4f}.",
                        flush=True,
                        end="",
                    )
                    sleep(1)
            except KeyboardInterrupt:
                print("\nkeyboard interrupt", flush=True)
                worker.alive = False
                # User interrupt, don't validate.
                FLAGS.validate = False
        finally:
            env.close()

        if FLAGS.validate:
            FLAGS.env = "llvm-ic-v0"
            validate(["argv0", FLAGS.leaderboard_results])

    app.run(main)
