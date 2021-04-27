# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for CompilerGym tests."""
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest
from absl import flags as absl_flags

FLAGS = absl_flags.FLAGS

# Decorator to skip a test in the CI environment.
skip_on_ci = pytest.mark.skipif(os.environ.get("CI", "") != "", reason="Skip on CI")

# Decorator to mark a test as skipped if not on Linux.
linux_only = pytest.mark.skipif(
    not sys.platform.lower().startswith("linux"), reason="Linux only"
)

# Decorator to mark a test as skipped if not on macOS.
macos_only = pytest.mark.skipif(
    not sys.platform.lower().startswith("darwin"), reason="macOS only"
)

# Decorator to mark a test as skipped if not running under bazel.
bazel_only = pytest.mark.skipif(
    os.environ.get("TEST_WORKSPACE", "") == "", reason="bazel only"
)


@pytest.fixture(scope="function")
def tmpwd() -> Path:
    """A fixture that creates a temporary directory, changes to it, and yields the path."""
    with tempfile.TemporaryDirectory(prefix="compiler_gym-test-") as d:
        pwd = os.getcwd()
        try:
            os.chdir(d)
            yield Path(d)
        finally:
            os.chdir(pwd)


@pytest.fixture(scope="function")
def temporary_environ():
    """A fixture that allows you to modify os.environ without affecting other tests."""
    old_env = os.environ.copy()
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def set_command_line_flags(flags: List[str]):
    """Set the command line flags."""
    sys.argv = flags
    FLAGS.unparse_flags()
    FLAGS(flags)
