# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.util.commands."""
import subprocess

import pytest

from compiler_gym.util.commands import Popen, communicate
from tests.test_main import main


def test_communicate_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        with subprocess.Popen(["sleep", "60"]) as process:
            communicate(process, timeout=1)
    assert process.poll() is not None  # Process is dead.


def test_popen():
    with Popen(["echo"]) as process:
        communicate(process, timeout=60)
    assert process.poll() is not None  # Process is dead.


if __name__ == "__main__":
    main()
