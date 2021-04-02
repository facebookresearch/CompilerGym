# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for resolving a runfiles path."""
import os
from pathlib import Path

# NOTE(cummins): Moving this file may require updating this relative path.
_PACKAGE_ROOT = Path(os.path.join(os.path.dirname(__file__), "../../")).resolve(
    strict=True
)


def runfiles_path(relpath: str) -> Path:
    """Resolve the path to a runfiles data path.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    Use environment variable COMPILER_GYM_RUNFILES=/path/to/runfiles if running
    outside of bazel.

    :param relpath: The relative path within the runfiles tree.

    :return: An absolute path.
    """
    # There are three ways of determining a runfiles path:
    #   1. Set the COMPILER_GYM_RUNFILES environment variable.
    #   2. Using the rules_python library that is provided by bazel. This will
    #      fail if not being executed within a bazel sandbox.
    #   3. Computing the path relative to the location of this file. This is the
    #      fallback approach that is used for when the code has been installed
    #      by setuptools.
    runfiles_path = os.environ.get("COMPILER_GYM_RUNFILES")
    if runfiles_path:
        return Path(runfiles_path) / relpath
    else:
        try:
            from rules_python.python.runfiles import runfiles

            return Path(
                runfiles.Create().Rlocation(
                    "CompilerGym" if relpath == "." else f"CompilerGym/{relpath}"
                )
            )
        except (ModuleNotFoundError, TypeError):
            return _PACKAGE_ROOT / relpath


def site_data_path(relpath: str) -> Path:
    """Return a path within the site data directory.

    CompilerGym uses a directory to store persistent site data files in, such as benchmark datasets.
    The default location is :code:`~/.local/share/compiler_gym`. Set the environment variable
    :code:`$COMPILER_GYM_SITE_DATA` to override this default location.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    :param relpath: The relative path within the site data tree.

    :return: An absolute path.
    """
    # NOTE(cummins): This function has a matching implementation in the C++
    # sources, compiler_gym::service::getSiteDataPath(). Any change to behavior
    # here must be reflected in the C++ version.
    forced = os.environ.get("COMPILER_GYM_SITE_DATA")
    if forced:
        return Path(forced) / relpath
    elif os.environ.get("HOME"):
        return Path("~/.local/share/compiler_gym").expanduser() / relpath
    else:
        return Path("/tmp/compiler_gym/site_data") / relpath


def cache_path(relpath: str) -> Path:
    """Return a path within the cache directory.

    CompilerGym uses a directory to cache files in, such as downloaded content.
    The default location for this cache is :code:`~/.cache/compiler_gym`. Set
    the environment variable :code:`$COMPILER_GYM_CACHE` to override this
    default location.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    :param relpath: The relative path within the cache tree.

    :return: An absolute path.
    """
    forced = os.environ.get("COMPILER_GYM_CACHE")
    if forced:
        return Path(forced) / relpath
    elif os.environ.get("HOME"):
        return Path("~/.cache/compiler_gym").expanduser() / relpath
    else:
        return Path("/tmp/compiler_gym/cache") / relpath


def transient_cache_path(relpath: str) -> Path:
    """Return a path within the transient cache directory.

    The transient cache is a directory used to store files that do not need to
    persist beyond the lifetime of the current process. When available, the
    temporary filesystem :code:`/dev/shm` will be used. Else,
    :meth:`cache_path() <compiler_gym.cache_path>` is used as a fallback. Set
    the environment variable :code:`$COMPILER_GYM_TRANSIENT_CACHE` to override
    the default location.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    :param relpath: The relative path within the cache tree.

    :return: An absolute path.
    """
    forced = os.environ.get("COMPILER_GYM_TRANSIENT_CACHE")
    if forced:
        return Path(forced) / relpath
    elif Path("/dev/shm").is_dir():
        return Path("/dev/shm/compiler_gym") / relpath
    else:
        # Fallback to using the regular cache.
        return cache_path(relpath)
