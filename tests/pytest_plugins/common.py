# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for CompilerGym tests."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

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


@pytest.fixture(scope="function")
def tmpwd() -> Path:
    """A fixture that creates a tempory directory, changes to it, and yields the path."""
    with tempfile.TemporaryDirectory(prefix="compiler_gym-test-") as d:
        pwd = os.getcwd()
        try:
            os.chdir(d)
            yield Path(d)
        finally:
            os.chdir(pwd)
