# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end tests for //compiler_gym/bin:benchmarks."""
import pytest

from compiler_gym.bin.datasets import main
from compiler_gym.util.capture_output import capture_output
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main as _test_main


def run_main(*args):
    set_command_line_flags(["argv"] + list(args))
    return main(["argv0"])


def test_llvm_summary():
    with capture_output() as out:
        run_main("--env=llvm-v0")

    assert "cbench-v1" in out.stdout


def test_datasets_is_deprecated():
    with pytest.deprecated_call(
        match="Command-line management of datasets is deprecated"
    ):
        run_main("--env=llvm-v0")


if __name__ == "__main__":
    _test_main()
