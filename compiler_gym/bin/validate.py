# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Validate environment states.

Example usage:

.. code-block::

    $ cat << EOF |
    benchmark,reward,walltime,commandline
    cBench-v0/crc32,0,1.2,opt  input.bc -o output.bc
    EOF
    python -m compiler_gym.bin.validate < results.csv --env=llvm-v0 --reward=IrInstructionCount

Use this script to validate environment states. Environment states are read from
stdin as a comma-separated list of benchmark names, walltimes, episode rewards,
and commandlines. Each state is validated by replaying the commandline and
validating that the reward matches the expected value. Further, some benchmarks
allow for validation of program semantics. When available, those additional
checks will be automatically run.

Input Format
------------

The correct format for generating input states can be generated using
:func:`env.state.to_csv() <compiler_gym.envs.CompilerEnvState.to_csv>`. The
input CSV must start with a header row. A valid header row can be generated
using
:func:`env.state.csv_header() <compiler_gym.envs.CompilerEnvState.csv_header>`.

Full example:

>>> env = gym.make("llvm-v0")
>>> env.reset()
>>> env.step(0)
>>> print(env.state.csv_header())
benchmark,reward,walltime,commandline
>>> print(env.state.to_csv())
benchmark://cBench-v0/rijndael,,20.53565216064453,opt -add-discriminators input.bc -o output.bc
%

Output Format
-------------

This script prints one line per input state. The order of input states is not
preserved. A successfully validated state has the format:

.. code-block::

    ✅  <benchmark_name>  <reproduced_reward>

Else if validation fails, the output is:

.. code-block::

    ❌  <benchmark_name>  <error_details>
"""
import csv
import sys

from absl import app, flags

import compiler_gym.util.flags.dataset  # Flag definition.
import compiler_gym.util.flags.nproc  # Flag definition.
from compiler_gym.envs.compiler_env import CompilerEnvState
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.validate import validate_states

FLAGS = flags.FLAGS


def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    data = sys.stdin.readlines()
    states = []
    for line in csv.DictReader(data):
        try:
            line["reward"] = float(line["reward"])
            states.append(CompilerEnvState(**line))
        except (TypeError, KeyError) as e:
            print(f"Failed to parse input: `{e}`", file=sys.stderr)
            sys.exit(1)

    error_count = 0
    for result in validate_states(
        env_from_flags, states, datasets=FLAGS.dataset, nproc=FLAGS.nproc
    ):
        print(result)
        if result.failed:
            error_count += 1

    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
