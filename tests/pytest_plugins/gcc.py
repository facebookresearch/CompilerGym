# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the GCC CompilerGym environments."""

import shutil
import subprocess
from functools import lru_cache
from typing import Iterable

import pytest

from tests.pytest_plugins.common import docker_is_available


def system_has_functional_gcc(gcc_path: str) -> bool:
    """Return whether there is a system GCC available."""
    try:
        stdout = subprocess.check_output(
            [gcc_path, "--version"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        # On some systems "gcc" may alias to a different compiler, so check for
        # the presence of the name "gcc" in the first line of output.
        return "gcc" in stdout.split("\n")[0].lower()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


@lru_cache
def system_gcc_is_available():
    return system_has_functional_gcc(shutil.which("gcc"))


@lru_cache
def gcc_bins() -> Iterable[str]:
    """Return a list of available GCCs."""
    if docker_is_available():
        yield "docker:gcc:11.2.0"
    system_gcc = shutil.which("gcc")
    if system_gcc and system_has_functional_gcc(system_gcc):
        yield system_gcc


def gcc_environment_is_supported() -> bool:
    """Return whether the requirements for the GCC environment are met."""
    return len(list(gcc_bins())) > 0


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
