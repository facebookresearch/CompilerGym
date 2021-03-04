# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for CompilerGym tests."""
import os
import tempfile
from pathlib import Path

import pytest

# Decorator to skip a test in the CI environment.
skip_on_ci = pytest.mark.skipif(os.environ.get("CI", "") != "", reason="Skip on CI")


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


@pytest.fixture(scope="function")
def temporary_environ():
    """A fixture that allows you to modify os.environ without affecting other tests."""
    old_env = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)
