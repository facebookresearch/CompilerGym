# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end tests for //compiler_gym/bin:benchmarks."""
import pytest
from absl import flags

from compiler_gym.bin.datasets import main
from compiler_gym.util.capture_output import capture_output
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def run_main(*args):
    FLAGS.unparse_flags()
    FLAGS(["argv"] + list(args))
    return main(["argv0"])


def test_llvm_summary():
    with capture_output() as out:
        run_main("--env=llvm-v0")

    assert "cbench-v1" in out.stdout


def test_llvm_download_invalid_protocol():
    invalid_url = "invalid://facebook.com"
    with pytest.raises(LookupError) as ctx:
        run_main("--env=llvm-v0", "--download", invalid_url)
    assert invalid_url in str(ctx.value)


if __name__ == "__main__":
    _test_main()
