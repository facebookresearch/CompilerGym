# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manage datasets of benchmarks.

.. code-block::

    $ python -m compiler_gym.bin.datasets --env=<env> [command...]

Where :code:`command` is one of :code:`--download=<dataset...>`,
:code:`--activate=<dataset...>`, :code:`--deactivate=<dataset...>`,
and :code:`--delete=<dataset...>`.


Listing installed datasets
--------------------------

If run with no arguments, this command shows an overview of the datasets that
are activate, inactive, and available to download. For example:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0
    llvm-v0 benchmarks site dir: /home/user/.local/share/compiler_gym/llvm/10.0.0/bitcode_benchmarks

    +-------------------+--------------+-----------------+----------------+
    | Active Datasets   | License      |   #. Benchmarks | Size on disk   |
    +===================+==============+=================+================+
    | cBench-v0         | BSD 3-Clause |              23 | 7.2 MB         |
    +-------------------+--------------+-----------------+----------------+
    | Total             |              |              23 | 7.2 MB         |
    +-------------------+--------------+-----------------+----------------+
    These benchmarks are ready for use. Deactivate them using `--deactivate=<name>`.

    +---------------------+-----------+-----------------+----------------+
    | Inactive Datasets   | License   |   #. Benchmarks | Size on disk   |
    +=====================+===========+=================+================+
    | Total               |           |               0 | 0 Bytes        |
    +---------------------+-----------+-----------------+----------------+
    These benchmarks may be activated using `--activate=<name>`.

    +------------------------+---------------------------------+-----------------+----------------+
    | Downloadable Dataset   | License                         | #. Benchmarks   | Size on disk   |
    +========================+=================================+=================+================+
    | blas-v0                | BSD 3-Clause                    | 300             | 4.0 MB         |
    +------------------------+---------------------------------+-----------------+----------------+
    | polybench-v0           | BSD 3-Clause                    | 27              | 162.6 kB       |
    +------------------------+---------------------------------+-----------------+----------------+
    These benchmarks may be installed using `--download=<name> --activate=<name>`.


Downloading datasets
--------------------

Use :code:`--download` to download a dataset from the list of available datasets:

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

:code:`--download` accepts the URL of any :code:`.tar.bz2` file to support custom datasets:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --download=https://example.com/dataset.tar.bz2

Or use the :code:`file:///` URI to install a local archive file:

.. code-block::

    $ python -m compiler_gym.bin.benchmarks --env=llvm-v0 --download=file:////tmp/dataset.tar.bz2

The list of datasets that are available to download may be extended by calling
:meth:`CompilerEnv.register_dataset() <compiler_gym.envs.CompilerEnv.register_dataset>`
on a :code:`CompilerEnv` instance.

To programmatically download datasets, see
:meth:`CompilerEnv.require_dataset() <compiler_gym.envs.CompilerEnv.require_dataset>`.

Activating and deactivating datasets
------------------------------------

Datasets have two states: active and inactive. An inactive dataset still exists
locally on the filesystem, but is excluded from use by CompilerGym environments.
This be useful if you have many datasets downloaded and you would to limit the
benchmarks that can be selected randomly by an environment.

Activate or deactivate datasets using the :code:`--activate` and :code:`--deactivate`
flags, respectively:

.. code-block::

    $ python -m comiler_gym.bin.benchmarks --env=llvm-v0 --activate=npb-v0,github-v0 --deactivate=cbench

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
import os
import sys
from pathlib import Path
from typing import Tuple

import fasteners
import humanize
from absl import app, flags

from compiler_gym.datasets.dataset import Dataset, activate, deactivate, delete, require
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.tabulate import tabulate

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


def get_count_and_size_of_directory_contents(root: Path) -> Tuple[int, int]:
    """Return the number of files and combined size of a directory."""
    count, size = 0, 0
    for root, _, files in os.walk(str(root)):
        count += len(files)
        size += sum(os.path.getsize(f"{root}/{file}") for file in files)
    return count, size


def enumerate_directory(name: str, path: Path):
    rows = []
    for path in path.iterdir():
        if not path.is_file() or not path.name.endswith(".json"):
            continue
        dataset = Dataset.from_json_file(path)
        rows.append(
            (dataset.name, dataset.license, dataset.file_count, dataset.size_bytes)
        )
    rows.append(("Total", "", sum(r[2] for r in rows), sum(r[3] for r in rows)))
    return tabulate(
        [(n, l, humanize.intcomma(f), humanize.naturalsize(s)) for n, l, f, s in rows],
        headers=(name, "License", "#. Benchmarks", "Size on disk"),
    )


def main(argv):
    """Main entry point."""
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    env = env_from_flags()
    try:
        if not env.datasets_site_path:
            raise app.UsageError("Environment has no benchmarks site path")

        env.datasets_site_path.mkdir(parents=True, exist_ok=True)
        env.inactive_datasets_site_path.mkdir(parents=True, exist_ok=True)

        invalidated_manifest = False

        for name_or_url in FLAGS.download:
            require(env, name_or_url)

        if FLAGS.download_all:
            for dataset in env.available_datasets:
                require(env, dataset)

        for name in FLAGS.activate:
            activate(env, name)
            invalidated_manifest = True

        if FLAGS.activate_all:
            for path in env.inactive_datasets_site_path.iterdir():
                activate(env, path.name)
            invalidated_manifest = True

        for name in FLAGS.deactivate:
            deactivate(env, name)
            invalidated_manifest = True

        if FLAGS.deactivate_all:
            for path in env.datasets_site_path.iterdir():
                deactivate(env, path.name)
            invalidated_manifest = True

        for name in FLAGS.delete:
            delete(env, name)

        if invalidated_manifest:
            env.make_manifest_file()

        print(f"{env.spec.id} benchmarks site dir: {env.datasets_site_path}")
        print()
        print(
            enumerate_directory("Active Datasets", env.datasets_site_path),
        )
        print(
            "These benchmarks are ready for use. Deactivate them using `--deactivate=<name>`."
        )
        print()
        print(enumerate_directory("Inactive Datasets", env.inactive_datasets_site_path))
        print("These benchmarks may be activated using `--activate=<name>`.")
        print()
        print(
            tabulate(
                sorted(
                    [
                        (
                            d.name,
                            d.license,
                            humanize.intcomma(d.file_count),
                            humanize.naturalsize(d.size_bytes),
                        )
                        for d in env.available_datasets.values()
                    ]
                ),
                headers=(
                    "Downloadable Dataset",
                    "License",
                    "#. Benchmarks",
                    "Size on disk",
                ),
            )
        )
        print(
            "These benchmarks may be installed using `--download=<name> --activate=<name>`."
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        app.run(main)
    except (ValueError, OSError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
