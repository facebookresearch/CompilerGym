# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This program lists the capabilities of CompilerGym services.

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
example, query a service running at :code:`localhost:8080` using:

.. code-block::

    $ python -m compiler_gym.bin.service --service=localhost:8080

To query the capability of a binary that implements the RPC service interface,
use the :code:`--local_service_binary` flag:

.. code-block::

    $ python -m compiler_gym.bin.service --local_service_binary=/path/to/service/binary
"""
from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.spaces import Commandline
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.tabulate import tabulate
from compiler_gym.util.truncate import truncate

flags.DEFINE_integer(
    "heading_level",
    1,
    "The base level for generated markdown headers, in the range [1,4].",
)
FLAGS = flags.FLAGS


def header(message: str, level: int):
    prefix = "#" * level
    return f"\n\n{prefix} {message}\n"


def print_service_capabilities(env: CompilerEnv, base_heading_level: int = 1):
    """Discover and print the capabilities of a CompilerGym service.

    :param env: An environment.
    """
    print(header(f"CompilerGym Service `{env.service}`", base_heading_level).strip())
    print(header("Programs", base_heading_level + 1))
    print(
        tabulate(
            [(p,) for p in sorted(env.benchmarks)],
            headers=("Benchmark",),
        )
    )
    print(header("Observation Spaces", base_heading_level + 1))
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
    print(header("Reward Spaces", base_heading_level + 1))
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

    print(header("Action Spaces", base_heading_level + 1).rstrip())
    for action_space in env.action_spaces:
        print(
            header(
                f"`{action_space.name}` ({type(action_space).__name__})",
                base_heading_level + 2,
            )
        )
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
        else:
            table = tabulate(
                [(a,) for a in sorted(action_space.names)],
                headers=("Action",),
            )
        print(table)


def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"
    assert 0 < FLAGS.heading_level <= 4, "--heading_level must be in range [1,4]"

    env = env_from_flags()
    try:
        print_service_capabilities(env, base_heading_level=FLAGS.heading_level)
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
