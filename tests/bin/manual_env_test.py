# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for //compiler_gym/bin:manual_env."""
import pytest
from absl import app, flags

from compiler_gym.bin.manual_env import main
from compiler_gym.util.capture_output import capture_output
from tests.test_main import main as _test_main

FLAGS = flags.FLAGS


def test_unrecognized_flags():
    FLAGS.unparse_flags()
    with pytest.raises(app.UsageError) as ctx:
        main(["argv0", "unknown-option"])
    assert str(ctx.value) == "Unknown command line arguments: ['unknown-option']"


def test_missing_required_flag():
    with pytest.raises(app.UsageError) as ctx:
        main(["argv0"])
    assert str(ctx.value) == "Neither --env or --local_service_binary is set"


def test_ls_env():
    with capture_output() as out:
        try:
            main(["argv0", "--ls_env"])
        except SystemExit:
            pass  # Expected behaviour is to call sys.exit().
    assert "llvm-" in out.stdout


if __name__ == "__main__":
    _test_main()
