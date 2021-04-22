# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end tests for //compiler_gym/bin:benchmarks."""
import pytest

from compiler_gym.bin.datasets import main
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main as _test_main


def run_main(*args):
    set_command_line_flags(["argv"] + list(args))
    return main(["argv0"])


def test_llvm_download_invalid_protocol():
    invalid_url = "invalid://facebook.com"
    with pytest.raises(OSError) as ctx:
        run_main("--env=llvm-v0", "--download", invalid_url)
    assert invalid_url in str(ctx.value)


if __name__ == "__main__":
    _test_main()
