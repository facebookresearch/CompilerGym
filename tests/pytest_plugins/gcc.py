# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the GCC CompilerGym environments."""

import subprocess
from functools import lru_cache
from typing import Iterable

import pytest

from tests.pytest_plugins.common import docker_is_available


@lru_cache(maxsize=2)
def system_gcc_is_available() -> bool:
    """Return whether there is a system GCC available."""
    try:
        stdout = subprocess.check_output(
            ["gcc", "--version"], universal_newlines=True, stderr=subprocess.DEVNULL
        )
        # On some systems "gcc" may alias to a different compiler, so check for
        # the presence of the name "gcc" in the first line of output.
        return "gcc" in stdout.split("\n")[0].lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def system_gcc_path() -> str:
    """Return the path of the system GCC as a string."""
    return subprocess.check_output(
        ["which", "gcc"], universal_newlines=True, stderr=subprocess.DEVNULL
    ).strip()


def gcc_environment_is_supported() -> bool:
    """Return whether the requirements for the GCC environment are met."""
    return docker_is_available() or system_gcc_is_available()


def gcc_bins() -> Iterable[str]:
    """Return a list of available GCCs."""
    if docker_is_available():
        yield "docker:gcc:11.2.0"
    if system_gcc_is_available():
        yield system_gcc_path()


@pytest.fixture(scope="module", params=gcc_bins())
def gcc_bin(request) -> str:
    return request.param


# Decorator to skip a test if GCC environment is not supported.
with_gcc_support = pytest.mark.skipif(
    not gcc_environment_is_supported(), reason="Docker is not available"
)

# Decorator to skip a test if GCC environment is supported.
without_gcc_support = pytest.mark.skipif(
    gcc_environment_is_supported(), reason="Docker is not available"
)

# Decorator to skip a test if system GCC is not availbale.
with_system_gcc = pytest.mark.skipif(
    not system_gcc_is_available(), reason="GCC is not available"
)

# Decorator to skip a test if system GCC is availbale.
without_system_gcc = pytest.mark.skipif(
    system_gcc_is_available(), reason="GCC is available"
)
