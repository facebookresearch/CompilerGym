# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:validate."""
from io import StringIO

import pytest
from absl import flags

from compiler_gym.bin.validate import main
from compiler_gym.util.capture_output import capture_output
from tests.test_main import main as _test_main


def test_okay_llvm_result(monkeypatch):
    input = """
benchmark,reward,commandline,walltime
benchmark://cBench-v0/dijkstra,0,opt  input.bc -o output.bc,0.3
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(input))

    with capture_output() as out:
        main(["argv0"])

    assert out.stdout == ("✅  benchmark://cBench-v0/dijkstra  0.0000\n")
    assert not out.stderr


def test_invalid_reward_llvm_result(monkeypatch):
    input = """
benchmark,reward,commandline,walltime
benchmark://cBench-v0/dijkstra,0.5,opt  input.bc -o output.bc,0.3
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(input))
    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0"])

    assert out.stdout == (
        "❌  benchmark://cBench-v0/dijkstra  Expected reward 0.5000 but received reward 0.0000\n"
    )
    assert not out.stderr


def test_invalid_csv_format(monkeypatch):
    input = "invalid\ncsv\nformat"
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(input))

    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0"])

    assert "Failed to parse input:" in out.stderr


if __name__ == "__main__":
    _test_main()
