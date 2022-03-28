# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This program can be used to query and run the CompilerGym services.

Listing available environments
------------------------------

List the environments that are available using:

.. code-block::

    $ python -m compiler_gym.bin.service --ls_env

Querying the capabilities of a service
--------------------------------------

Query the capabilities of a service using:

.. code-block::

    $ python -m compiler_gym.bin.service --env=<env>

For example:

.. code-block::

    $ python -m compiler_gym.bin.service --env=llvm-v0

    Datasets
    --------

    +----------------------------+--------------------------+------------------------------+
    | Dataset                    | Num. Benchmarks [#f1]_   | Description                  |
    +============================+==========================+==============================+
    | benchmark://anghabench-v0  | 1,042,976                | Compile-only C/C++ functions |
    +----------------------------+--------------------------+------------------------------+
    | benchmark://blas-v0        | 300                      | Basic linear algebra kernels |
    +----------------------------+--------------------------+------------------------------+
    ...

    Observation Spaces
    ------------------

    +--------------------------+----------------------------------------------+
    | Observation space        | Shape                                        |
    +==========================+==============================================+
    | Autophase                | `Box(0, 9223372036854775807, (56,), int64)`  |
    +--------------------------+----------------------------------------------+
    | AutophaseDict            | `Dict(ArgsPhi:int<0,inf>, BB03Phi:int<0,...` |
    +--------------------------+----------------------------------------------+
    | BitcodeFile              | `str_list<>[0,4096.0])`                      |
    +--------------------------+----------------------------------------------+
    ...

The output is tabular summaries of the environment's datasets, observation
spaces, reward spaces, and action spaces, using reStructuredText syntax
(https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#tables).

To query the capabilities of an unmanaged service, use :code:`--service`. For
example, query a service running the :code:`llvm-v0` environment at
:code:`localhost:8080` using:

.. code-block::

    $ python -m compiler_gym.bin.service --env=llvm-v0 --service=localhost:8080

To query the capability of a binary that implements the RPC service interface,
use the :code:`--local_service_binary` flag:

.. code-block::

    $ python -m compiler_gym.bin.service --env=llvm-v0 --local_service_binary=/path/to/service/binary


Running a Service
-----------------

This module can also be used to launch a service that can then be connected to
by other environments. Start a service by specifying a port number using:

.. code-block::

    $ python -m compiler_gym.bin.service --env=llvm-v0 --run_on_port=7777

Environment can connect to this service by passing the :code:`<hostname>:<port>`
address during environment initialization time. For example, in python:

    >>> env = compiler_gym.make("llvm-v0", service="localhost:7777")

Or at the command line:

.. code-block::

    $ python -m compiler_gym.bin.random_search --env=llvm-v0 --service=localhost:7777
"""
import signal
import sys
from typing import Iterable

import gym
from absl import app, flags

from compiler_gym.datasets import Dataset
from compiler_gym.envs import CompilerEnv
from compiler_gym.service.connection import ConnectionOpts
from compiler_gym.spaces import Commandline, NamedDiscrete
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.truncate import truncate

flags.DEFINE_string(
    "heading_underline_char",
    "-",
    "The character to repeat to underline headings.",
)
flags.DEFINE_integer(
    "run_on_port",
    None,
    "Is specified, serve an instance of the service on the requested port. "
    "This never terminates.",
)
FLAGS = flags.FLAGS


def header(message: str):
    underline = FLAGS.heading_underline_char * (
        len(message) // len(FLAGS.heading_underline_char)
    )
    return f"\n\n{message}\n{underline}\n"


def shape2str(shape, n: int = 80):
    string = str(shape)
    if len(string) > n:
        return f"`{string[:n-4]}` ..."
    return f"`{string}`"


def summarize_datasets(datasets: Iterable[Dataset]) -> str:
    rows = []
    # Override the default iteration order of datasets.
    for dataset in sorted(datasets, key=lambda d: d.name):
        # Raw numeric values here, formatted below.
        description = truncate(dataset.description, max_line_len=60)
        links = ", ".join(
            f"`{name} <{url}>`__" for name, url in sorted(dataset.references.items())
        )
        if links:
            description = f"{description} [{links}]"
        rows.append(
            (
                dataset.name,
                dataset.size,
                description,
                dataset.validatable,
            )
        )
    rows.append(("Total", sum(r[1] for r in rows), "", ""))
    return (
        tabulate(
            [
                (
                    n,
                    # A size of zero means infinite.
                    f"{f:,d}" if f > 0 else "∞",
                    l,
                    v,
                )
                for n, f, l, v in rows
            ],
            headers=(
                "Dataset",
                "Num. Benchmarks [#f1]_",
                "Description",
                "Validatable [#f2]_",
            ),
        )
        + f"""

.. [#f1] Values obtained on {sys.platform}. Datasets are platform-specific.
.. [#f2] A **validatable** dataset is one where the behavior of the benchmarks
         can be checked by compiling the programs to binaries and executing
         them. If the benchmarks crash, or are found to have different behavior,
         then validation fails. This type of validation is used to check that
         the compiler has not broken the semantics of the program.
         See :mod:`compiler_gym.bin.validate`.
"""
    )


def print_service_capabilities(env: CompilerEnv):
    """Discover and print the capabilities of a CompilerGym service.

    :param env: An environment.
    """
    print(header("Datasets"))
    print(
        summarize_datasets(env.datasets),
    )
    print(header("Observation Spaces"))
    print(
        tabulate(
            sorted(
                [
                    (space, f"`{truncate(shape.space, max_line_len=80)}`")
                    for space, shape in env.observation.spaces.items()
                ]
            ),
            headers=("Observation space", "Shape"),
        )
    )
    print(header("Reward Spaces"))
    print(
        tabulate(
            [
                (
                    name,
                    space.range,
                    space.success_threshold,
                    "Yes" if space.deterministic else "No",
                    "Yes" if space.platform_dependent else "No",
                )
                for name, space in sorted(env.reward.spaces.items())
            ],
            headers=(
                "Reward space",
                "Range",
                "Success threshold",
                "Deterministic?",
                "Platform dependent?",
            ),
        )
    )

    for action_space in env.action_spaces:
        print(header(f"{action_space.name} Action Space"))
        # Special handling for commandline action spaces to print additional
        # information.
        if isinstance(action_space, Commandline):
            table = tabulate(
                [
                    (f"`{n}`", d)
                    for n, d in zip(
                        action_space.names,
                        action_space.descriptions,
                    )
                ],
                headers=("Action", "Description"),
            )
            print(table)
        elif isinstance(action_space, NamedDiscrete):
            table = tabulate(
                [(a,) for a in sorted(action_space.names)],
                headers=("Action",),
            )
            print(table)
        else:
            raise NotImplementedError(
                "Only Commandline and NamedDiscrete are supported."
            )


def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    if FLAGS.run_on_port:
        assert FLAGS.env, "Must specify an --env to run"
        settings = ConnectionOpts(script_args=["--port", str(FLAGS.run_on_port)])
        with gym.make(FLAGS.env, connection_settings=settings) as env:
            print(
                f"=== Started a service on port {FLAGS.run_on_port}. Use C-c to terminate. ==="
            )
            signal.pause()

    with env_from_flags() as env:
        print_service_capabilities(env)


if __name__ == "__main__":
    app.run(main)
