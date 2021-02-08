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
    python -m compiler_gym.bin.validate < results.csv --env=llvm-ic-v0

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
import re
import sys
from typing import Iterator

import numpy as np
from absl import app, flags

import compiler_gym.util.flags.dataset  # noqa Flag definition.
import compiler_gym.util.flags.nproc  # noqa Flag definition.
from compiler_gym import ValidationResult
from compiler_gym.envs.compiler_env import CompilerEnvState
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.statistics import geometric_mean
from compiler_gym.validate import validate_states

flags.DEFINE_boolean(
    "inorder",
    False,
    "Whether to print results in the order they are provided. "
    "The default is to print results as soon as they are available.",
)
FLAGS = flags.FLAGS


def read_states_from_stdin() -> Iterator[CompilerEnvState]:
    """Read the CSV states from stdin."""
    data = sys.stdin.readlines()
    for line in csv.DictReader(data):
        try:
            line["reward"] = float(line["reward"]) if line.get("reward") else None
            line["walltime"] = float(line["walltime"]) if line.get("walltime") else None
            yield CompilerEnvState(**line)
        except (TypeError, KeyError) as e:
            print(f"Failed to parse input: `{e}`", file=sys.stderr)
            sys.exit(1)


def state_name(state: CompilerEnvState) -> str:
    """Get the string name for a state."""
    return re.sub(r"^benchmark://", "", state.benchmark)


def to_string(result: ValidationResult, name_col_width: int) -> str:
    """Format a validation result for printing."""
    name = state_name(result.state)

    if result.failed:
        msg = ", ".join(result.error_details.strip().split("\n"))
        return f"❌  {name}  {msg}"
    elif result.state.reward is None:
        return f"✅  {name}"
    else:
        return f"✅  {name:<{name_col_width}}  {result.state.reward:9.4f}"


def arithmetic_mean(values):
    """Zero-length-safe arithmetic mean."""
    if not values:
        return 0
    return sum(values) / len(values)


def stdev(values):
    """Zero-length-safe standard deviation."""
    return np.std(values or [0])


def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    # Parse the input states from the user.
    states = list(read_states_from_stdin())

    # Send the states off for validation
    validation_results = validate_states(
        env_from_flags,
        states,
        datasets=FLAGS.dataset,
        nproc=FLAGS.nproc,
        inorder=FLAGS.inorder,
    )

    # Determine the name of the reward space.
    env = env_from_flags()
    try:
        if env.reward_space:
            gmean_name = f"Geometric mean {env.reward_space.id}"
        else:
            gmean_name = "Geometric mean"
    finally:
        env.close()

    # Determine the maximum column width required for printing tabular output.
    max_state_name_length = max(
        len(s)
        for s in [state_name(s) for s in states]
        + [
            "Mean inference walltime",
            gmean_name,
        ]
    )
    name_col_width = min(max_state_name_length + 2, 78)

    error_count = 0
    rewards = []
    walltimes = []

    for result in validation_results:
        print(to_string(result, name_col_width))
        if result.failed:
            error_count += 1
        elif result.reward_validated and not result.reward_validation_failed:
            rewards.append(result.state.reward)
            walltimes.append(result.state.walltime)

    # Print a summary footer.
    print("----", "-" * name_col_width, "-----------", sep="")
    print(f"Number of validated results: {emph(len(walltimes))} of {len(states)}")
    walltime_mean = f"{arithmetic_mean(walltimes):.3f}s"
    walltime_std = f"{stdev(walltimes):.3f}s"
    print(
        f"Mean inference walltime: {emph(walltime_mean)} sec / benchmark "
        f"(std: {emph(walltime_std)})"
    )
    reward_gmean = f"{geometric_mean(rewards):.3f}"
    reward_std = f"{stdev(rewards):.3f}"
    print(f"{gmean_name}: {emph(reward_gmean)} (std: {emph(reward_std)})")

    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
