# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a parallelized random search of an environment's action space.

.. code-block::

    $ python -m compiler_gym.bin.random_search --env=<env> --benchmark=<name> [--runtime=<sec>]

This program runs a simple random agent on the action space of a single
benchmark. The best reward, and sequence of actions that produced this, are
logged to file.

For example, run a random search of the LLVM instruction count optimization
space on a Dijkstra benchmark for 60 seconds using:

.. code-block::

    $ python -m compiler_gym.bin.random_search --env=llvm-ic-v0 --benchmark=cbench-v1/dijkstra --runtime=60
    Started 16 worker threads for benchmark benchmark://cbench-v1/dijkstra (410 instructions) using reward IrInstructionCountOz.
    === Running for a minute ===
    Runtime: a minute. Num steps: 470,407 (7,780 / sec). Num episodes: 4,616 (76 / sec). Num restarts: 0.
    Best reward: 101.59% (96 passes, found after 35 seconds)
    Ending jobs ... done
    Step [000 / 096]: reward=0.621951
    Step [001 / 096]: reward=0.621951, change=0.000000, action=AlwaysInlinerLegacyPass
    ...
    Step [094 / 096]: reward=1.007905, change=0.066946, action=CfgsimplificationPass
    Step [095 / 096]: reward=1.007905, change=0.000000, action=LoopVersioningPass
    Step [096 / 096]: reward=1.015936, change=0.008031, action=NewGvnpass

Search strategy
---------------

At each step, the agent selects an action randomly and records the
reward. After a number of steps without improving reward (the "patience" of the
agent), the agent terminates, and the environment resets. The number of steps
to take without making progress can be configured using the
:code:`--patience=<num>` flag.

Use :code:`--runtime` to limit the total runtime of the search. If not provided,
the search will run indefinitely. Use :code:`C-c` to cancel an in-progress
search.

Execution strategy
------------------

The results of the search are logged to files. Control the location of these
logs using the :code:`--output_dir=/path` flag.

Multiple agents are run in parallel. By default, the number of agents is equal
to the number of processors on the host machine. Set a different value using
:code:`--nproc`.
"""
import sys
from pathlib import Path

from absl import app, flags

import compiler_gym.util.flags.nproc  # noqa Flag definition.
import compiler_gym.util.flags.output_dir  # noqa Flag definition.
from compiler_gym.random_search import random_search
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_boolean("ls_reward", False, "List the available reward spaces and exit.")
flags.DEFINE_integer(
    "patience",
    0,
    "The number of steps that a random agent makes without improvement before terminating. "
    "If 0, use the size of the action space for the patience value.",
)
flags.DEFINE_float("runtime", None, "If set, limit the search to this many seconds.")
flags.DEFINE_boolean(
    "skip_done",
    False,
    "If set, don't overwrite existing experimental results.",
)
flags.DEFINE_float(
    "fail_threshold",
    None,
    "If set, define a minimum threshold for reward. The script will exit with return code 1 "
    "if this threshold is not reached.",
)
FLAGS = flags.FLAGS


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    if FLAGS.ls_reward:
        env = env_from_flags()
        print("\n".join(sorted(env.reward.indices.keys())))
        env.close()
        return

    assert FLAGS.patience >= 0, "--patience must be >= 0"

    def make_env():
        return env_from_flags(benchmark=benchmark_from_flags())

    env = make_env()
    try:
        env.reset()
    finally:
        env.close()

    best_reward, _ = random_search(
        make_env=make_env,
        outdir=Path(FLAGS.output_dir) if FLAGS.output_dir else None,
        patience=FLAGS.patience,
        total_runtime=FLAGS.runtime,
        nproc=FLAGS.nproc,
        skip_done=FLAGS.skip_done,
    )

    # Exit with error if --fail_threshold was set and the best reward does not
    # meet this value.
    if FLAGS.fail_threshold is not None and best_reward < FLAGS.fail_threshold:
        print(
            f"Best reward {best_reward:.3f} below threshold of {FLAGS.fail_threshold}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
