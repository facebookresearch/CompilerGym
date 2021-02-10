# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for CompilerGym tests."""
import os

import pytest

# Decorator to skip a test in the CI environment.
skip_on_ci = pytest.mark.skipif(os.environ.get("CI", "") != "", reason="Skip on CI")
