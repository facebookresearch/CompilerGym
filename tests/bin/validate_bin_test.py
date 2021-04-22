# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:validate."""
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from compiler_gym.bin.validate import main
from compiler_gym.util.capture_output import capture_output
from tests.pytest_plugins.common import set_command_line_flags, skip_on_ci
from tests.test_main import main as _test_main


def test_okay_llvm_result(monkeypatch):
    stdin = """
benchmark,reward,commandline,walltime
benchmark://cBench-v1/crc32,0,opt  input.bc -o output.bc,0.3
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert "✅  cBench-v1/crc32 " in out.stdout
    assert not out.stderr


def test_okay_llvm_result_file_input():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.csv"
        with open(str(path), "w") as f:
            f.write(
                """
benchmark,reward,commandline,walltime
benchmark://cBench-v1/crc32,0,opt  input.bc -o output.bc,0.3
""".strip()
            )
        set_command_line_flags(["argv0", "--env=llvm-ic-v0"])

        with capture_output() as out:
            main(["argv0", str(path)])

    assert "✅  cBench-v1/crc32 " in out.stdout
    assert not out.stderr


def test_no_input(monkeypatch):
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(""))

    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert "No inputs to validate" in out.stderr


def test_invalid_reward_llvm_result(monkeypatch):
    stdin = """
benchmark,reward,commandline,walltime
benchmark://cBench-v1/crc32,0.5,opt  input.bc -o output.bc,0.3
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))
    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert (
        "❌  cBench-v1/crc32  Expected reward 0.5 but received reward 0.0\n"
        in out.stdout
    )
    assert not out.stderr


def test_invalid_csv_format(monkeypatch):
    stdin = "invalid\ncsv\nformat"
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert "Failed to parse input:" in out.stderr


@skip_on_ci
def test_multiple_valid_inputs(monkeypatch):
    stdin = """
benchmark,reward,walltime,commandline
benchmark://cBench-v1/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/crc32,,0,opt  input.bc -o output.bc
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == 3  # Every benchmark passed.


@skip_on_ci
def test_validate_cBench_null_options(monkeypatch):
    stdin = """
benchmark,reward,walltime,commandline
benchmark://cBench-v1/gsm,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/lame,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/stringsearch,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/ghostscript,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/qsort,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/sha,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/ispell,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/blowfish,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/adpcm,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/tiffdither,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/bzip2,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/stringsearch2,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/bitcount,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/jpeg-d,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/jpeg-c,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/dijkstra,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/rijndael,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/patricia,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/tiff2rgba,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/tiff2bw,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/tiffmedian,,0,opt  input.bc -o output.bc
benchmark://cBench-v1/susan,,0,opt  input.bc -o output.bc
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == 23  # Every benchmark passed.


if __name__ == "__main__":
    _test_main()
