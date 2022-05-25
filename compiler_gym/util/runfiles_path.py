# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Module for resolving a runfiles path."""
import os
from datetime import datetime
from getpass import getuser
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Optional

# NOTE(cummins): Moving this file may require updating this relative path.
_PACKAGE_ROOT = Path(os.path.join(os.path.dirname(__file__), "../../")).resolve(
    strict=True
)

_CREATE_LOGGING_DIR_LOCK = Lock()


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

    CompilerGym uses a directory to store persistent site data files in, such as
    benchmark datasets. The default location is
    :code:`~/.local/share/compiler_gym`. Set the environment variable
    :code:`$COMPILER_GYM_SITE_DATA` to override this default location.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    Files in this directory are intended to be long lived (this is not a cache),
    but it is safe to delete this directory, so long as no CompilerGym
    environments are running.

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
        return Path(f"/tmp/compiler_gym_{getuser()}/site_data") / relpath


def cache_path(relpath: str) -> Path:
    """Return a path within the cache directory.

    CompilerGym uses a directory to cache files in, such as downloaded content.
    The default location for this cache is :code:`~/.local/cache/compiler_gym`.
    Set the environment variable :code:`$COMPILER_GYM_CACHE` to override this
    default location.

    It is safe to delete this directory, so long as no CompilerGym environments
    are running.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    :param relpath: The relative path within the cache tree.

    :return: An absolute path.
    """
    forced = os.environ.get("COMPILER_GYM_CACHE")
    if forced:
        return Path(forced) / relpath
    elif os.environ.get("HOME"):
        return Path("~/.local/cache/compiler_gym").expanduser() / relpath
    else:
        return Path(f"/tmp/compiler_gym_{getuser()}/cache") / relpath


def transient_cache_path(relpath: str) -> Path:
    """Return a path within the transient cache directory.

    The transient cache is a directory used to store files that do not need to
    persist beyond the lifetime of the current process. When available, the
    temporary filesystem :code:`/dev/shm` will be used. Else,
    :meth:`cache_path() <compiler_gym.cache_path>` is used as a fallback. Set
    the environment variable :code:`$COMPILER_GYM_TRANSIENT_CACHE` to override
    the default location.

    Files in this directory are not meant to outlive the lifespan of the
    CompilerGym environment that creates them. It is safe to delete this
    directory, so long as no CompilerGym environments are running.

    No checks are to made to ensure that the path, or the containing directory,
    exist.

    :param relpath: The relative path within the cache tree.

    :return: An absolute path.
    """
    forced = os.environ.get("COMPILER_GYM_TRANSIENT_CACHE")
    if forced:
        return Path(forced) / relpath
    elif Path("/dev/shm").is_dir():
        return Path(f"/dev/shm/compiler_gym_{getuser()}") / relpath
    else:
        # Fallback to using the regular cache.
        return cache_path(relpath)


def create_user_logs_dir(name: str, dir: Optional[Path] = None) -> Path:
    """Create a directory for writing logs to.

    Defaults to ~/logs/compiler_gym base directory, set the
    :code:`COMPILER_GYM_LOGS` environment variable to override this.

    Example use:

        >>> create_user_logs_dir("my_experiment")
        Path("~/logs/compiler_gym/my_experiment/2020-11-03T11:00:00")

    :param name: The grouping name for the logs.

    :return: A unique timestamped directory for logging. This directory exists.
    """
    base_dir = Path(
        os.environ.get("COMPILER_GYM_LOGS", dir or "~/logs/compiler_gym")
    ).expanduser()
    group_dir = base_dir / name

    with _CREATE_LOGGING_DIR_LOCK:
        # Require that logging directory timestamps are unique by waiting until
        # a unique timestamp is generated.
        while True:
            now = datetime.now()
            subdirs = now.strftime("%Y-%m-%d/%H-%M-%S")

            logs_dir = group_dir / subdirs
            if logs_dir.is_dir():
                sleep(0.3)
                continue

            logs_dir.mkdir(parents=True, exist_ok=False)

            # Create a symlink to the "latest" logs results.
            if (group_dir / "latest").exists():
                os.unlink(group_dir / "latest")
            os.symlink(subdirs, group_dir / "latest")

            return logs_dir
