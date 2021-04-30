# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:validate."""
import tempfile
from io import StringIO
from pathlib import Path
from typing import List

import pytest

from compiler_gym.bin.validate import main
from compiler_gym.util.capture_output import capture_output
from tests.pytest_plugins.common import set_command_line_flags, skip_on_ci
from tests.test_main import main as _test_main


def test_okay_llvm_result(monkeypatch):
    stdin = """
benchmark,reward,commandline,walltime
benchmark://cbench-v1/crc32,0,opt  input.bc -o output.bc,0.3
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert "✅  cbench-v1/crc32 " in out.stdout
    assert not out.stderr


def test_okay_llvm_result_file_input():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.csv"
        with open(str(path), "w") as f:
            f.write(
                """
benchmark,reward,commandline,walltime
benchmark://cbench-v1/crc32,0,opt  input.bc -o output.bc,0.3
""".strip()
            )
        set_command_line_flags(["argv0", "--env=llvm-ic-v0"])

        with capture_output() as out:
            main(["argv0", str(path)])

    assert "✅  cbench-v1/crc32 " in out.stdout
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
benchmark://cbench-v1/crc32,0.5,opt  input.bc -o output.bc,0.3
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-ic-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))
    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert (
        "❌  cbench-v1/crc32  Expected reward 0.5 but received reward 0.0\n"
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

    assert "Expected 4 columns in the first row of CSV" in out.stderr


@skip_on_ci
def test_multiple_valid_inputs(monkeypatch):
    stdin = """
benchmark,reward,walltime,commandline
benchmark://cbench-v1/crc32,,0,opt  input.bc -o output.bc
benchmark://cbench-v1/crc32,,0,opt  input.bc -o output.bc
benchmark://cbench-v1/crc32,,0,opt  input.bc -o output.bc
""".strip()
    set_command_line_flags(["argv0", "--env=llvm-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == 3  # Every benchmark passed.


@skip_on_ci
@pytest.mark.parametrize(
    "benchmarks",
    [
        [
            "benchmark://cbench-v1/gsm",
            "benchmark://cbench-v1/lame",
            "benchmark://cbench-v1/stringsearch",
            "benchmark://cbench-v1/ghostscript",
        ],
        [
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/sha",
            "benchmark://cbench-v1/ispell",
            "benchmark://cbench-v1/blowfish",
        ],
        [
            "benchmark://cbench-v1/adpcm",
            "benchmark://cbench-v1/tiffdither",
            "benchmark://cbench-v1/bzip2",
            "benchmark://cbench-v1/stringsearch2",
        ],
        [
            "benchmark://cbench-v1/bitcount",
            "benchmark://cbench-v1/jpeg-d",
            "benchmark://cbench-v1/jpeg-c",
            "benchmark://cbench-v1/dijkstra",
        ],
        [
            "benchmark://cbench-v1/rijndael",
            "benchmark://cbench-v1/patricia",
            "benchmark://cbench-v1/tiff2rgba",
            "benchmark://cbench-v1/crc32",
        ],
        [
            "benchmark://cbench-v1/tiff2bw",
            "benchmark://cbench-v1/tiffmedian",
            "benchmark://cbench-v1/susan",
        ],
    ],
)
def test_validate_cbench_null_options(monkeypatch, benchmarks: List[str]):
    stdin = "\n".join(
        [
            "benchmark,reward,walltime,commandline",
        ]
        + [f"{b},,0,opt  input.bc -o output.bc" for b in benchmarks]
    )
    set_command_line_flags(["argv0", "--env=llvm-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))
    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == len(benchmarks)  # Every benchmark passed.


if __name__ == "__main__":
    _test_main()
