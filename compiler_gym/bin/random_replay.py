# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Replay the best solution found from a random search.

.. code-block::

    $ python -m compiler_gym.bin.random_replay --env=llvm-ic-v0 --output_dir=/path/to/logs

Given a set of :mod:`compiler_gym.bin.random_search` logs generated from a
prior search, replay the best sequence of actions found and record the
incremental reward of each action.
"""
from pathlib import Path

from absl import app, flags

import compiler_gym.util.flags.output_dir  # noqa Flag definition.
from compiler_gym.random_replay import replay_actions_from_logs
from compiler_gym.util import logs
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

FLAGS = flags.FLAGS


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    output_dir = Path(FLAGS.output_dir).expanduser().resolve().absolute()
    assert (
        output_dir / logs.METADATA_NAME
    ).is_file(), f"Invalid --output_dir: {output_dir}"

    env = env_from_flags()
    benchmark = benchmark_from_flags()
    replay_actions_from_logs(env, output_dir, benchmark=benchmark)


if __name__ == "__main__":
    app.run(main)
