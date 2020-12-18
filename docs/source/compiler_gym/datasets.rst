compiler_gym.datasets
=====================

An instance of a CompilerGym environment uses a benchmark as the program being
optimized. Collections of benchmarks are packaged into datasets, storing
additional metadata such as the license, defined by the
:class:`Dataset <compiler_gym.datasets.Dataset>` class.

A simple filesystem-based scheme is used to manage datasets:

* Every top-level directory in an environment's site-data folder is
  treated as a "dataset".

* A benchmarks.inactive directory contains datasets that the user has
  downloaded, but are not used by the environment. Moving a directory
  from <site>/benchmarks to <site>/benchmarks.inactive means that the
  environment will no longer use it.

* Datasets can be packaged as .tar.bz2 archives and downloaded from
  the web or local filesystem. Environments may advertise a list of
  available datasets.

Datasets are packaged for each compiler and stored locally in the filesystem.
The filesystem location can be queries using
:attr:`CompilerEnv.datasets_site_path <compiler_gym.envs.CompilerEnv.datasets_site_path>`:

    >>> env = gym.make("llvm-v0")
    >>> env.datasets_site_path
    /home/user/.local/share/compiler_gym/llvm/10.0.0/bitcode_benchmarks

The :mod:`compiler_gym.bin.datasets` module can be used to download and manage
datasets for an environment.

.. automodule:: compiler_gym.datasets
   :members:
