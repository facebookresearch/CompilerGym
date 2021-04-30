# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manage datasets of benchmarks.

.. code-block::

    $ python -m compiler_gym.bin.datasets --env=<env> \
        [--download=<dataset...>] [--delete=<dataset...>]


Listing installed datasets
--------------------------

If run with no arguments, this command shows an overview of the datasets that
are activate, inactive, and available to download. For example:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0

    +-------------------+---------------------+-----------------+----------------+
    | Active Datasets   | Description         |   #. Benchmarks | Size on disk   |
    +===================+=====================+=================+================+
    | cbench-v1         | Runnable C programs |              23 | 10.1 MB        |
    +-------------------+---------------------+-----------------+----------------+
    | Total             |                     |              23 | 10.1 MB        |
    +-------------------+---------------------+-----------------+----------------+


Downloading datasets
--------------------

Use :code:`--download` to download a dataset from the list of available
datasets:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --download=npb-v0

After downloading, the dataset will be activated and the benchmarks will be
available to use by the environment.

    >>> import compiler_gym
    >>> import gym
    >>> env = gym.make("llvm-v0")
    >>> env.benchmark = "npb-v0"

The flag :code:`--download_all` can be used to download every available dataset:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --download_all

Or use the :code:`file:///` URI to install a local archive file:

.. code-block::

    $ python -m compiler_gym.bin.benchmarks --env=llvm-v0 --download=file:////tmp/dataset.tar.bz2


Activating and deactivating datasets
------------------------------------

Datasets have two states: active and inactive. An inactive dataset still exists
locally on the filesystem, but is excluded from use by CompilerGym environments.
This be useful if you have many datasets downloaded and you would to limit the
benchmarks that can be selected randomly by an environment.

Activate or deactivate datasets using the :code:`--activate` and
:code:`--deactivate` flags, respectively:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --activate=npb-v0,github-v0 --deactivate=cbench-v1

The :code:`--activate_all` and :code:`--deactivate_all` flags can be used as a
shortcut to activate or deactivate every downloaded:

.. code-block::

    # Activate all inactivate datasets:
    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --activate_all
    # Make all activate datasets inactive:
    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --deactivate_all

Deleting datasets
-----------------

To remove a dataset from the filesystem, use :code:`--delete`:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --delete=npb-v0

Once deleted, a dataset must be downloaded before it can be used again.

A :code:`--delete_all` flag can be used to delete all of the locally installed
datasets.
"""
import sys

from absl import app, flags
from deprecated.sphinx import deprecated

from compiler_gym.bin.service import summarize_datasets
from compiler_gym.datasets.dataset import activate, deactivate, delete
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_list(
    "download",
    [],
    "The name or URL of a dataset to download. Accepts a list of choices",
)
flags.DEFINE_list(
    "activate",
    [],
    "The names of one or more inactive datasets to activate. Accepts a list of choices",
)
flags.DEFINE_list(
    "deactivate",
    [],
    "The names of one or more active datasets to deactivate. Accepts a list of choices",
)
flags.DEFINE_list(
    "delete",
    [],
    "The names of one or more inactive dataset to delete. Accepts a list of choices",
)
flags.DEFINE_boolean("download_all", False, "Download all available datasets")
flags.DEFINE_boolean("activate_all", False, "Activate all inactive datasets")
flags.DEFINE_boolean("deactivate_all", False, "Deactivate all active datasets")
FLAGS = flags.FLAGS


@deprecated(
    version="0.1.8",
    reason=(
        "Command-line management of datasets is deprecated. Please use "
        ":mod:`compiler_gym.bin.service` to print a tabular overview of the "
        "available datasets. For management of datasets, use the "
        ":class:`env.datasets <compiler_gym.env>` property."
    ),
)
def main(argv):
    """Main entry point."""
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    env = env_from_flags()
    try:
        invalidated_manifest = False

        for name_or_url in FLAGS.download:
            env.datasets.install(name_or_url)

        if FLAGS.download_all:
            for dataset in env.datasets:
                dataset.install()

        for name in FLAGS.activate:
            activate(env, name)
            invalidated_manifest = True

        if FLAGS.activate_all:
            invalidated_manifest = True

        for name in FLAGS.deactivate:
            deactivate(env, name)
            invalidated_manifest = True

        if FLAGS.deactivate_all:
            invalidated_manifest = True

        for name in FLAGS.delete:
            delete(env, name)

        if invalidated_manifest:
            env.make_manifest_file()

        print(
            summarize_datasets(env.datasets),
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        app.run(main)
    except (ValueError, OSError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
