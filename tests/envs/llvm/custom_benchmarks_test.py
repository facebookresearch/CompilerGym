# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM benchmark handling."""
import tempfile
from pathlib import Path

import pytest

from compiler_gym.envs import CompilerEnv
from compiler_gym.service.proto import Benchmark, File
from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

EXAMPLE_BITCODE_FILE = runfiles_path("CompilerGym/third_party/cBench-v0/crc32.bc")


def test_reset_invalid_benchmark(env: CompilerEnv):
    invalid_benchmark = "an invalid benchmark"
    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=invalid_benchmark)

    assert str(ctx.value) == f'Unknown benchmark "{invalid_benchmark}"'


def test_invalid_benchmark_data(env: CompilerEnv):
    benchmark = Benchmark(
        uri="benchmark://new", program=File(contents="Invalid bitcode".encode("utf-8"))
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == 'Failed to parse LLVM bitcode: "benchmark://new"'


def test_invalid_benchmark_missing_file(env: CompilerEnv):
    benchmark = Benchmark(
        uri="benchmark://new",
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == "No program set"


def test_benchmark_path_not_found(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        benchmark = Benchmark(
            uri="benchmark://new", program=File(uri=f"file:///{tmpdir}/not_found")
        )

        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == f'File not found: "{tmpdir}/not_found"'


def test_benchmark_path_empty_file(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.bc").touch()

        benchmark = Benchmark(
            uri="benchmark://new", program=File(uri=f"file:///{tmpdir}/test.bc")
        )

        with pytest.raises(ValueError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == f'File is empty: "{tmpdir}/test.bc"'


def test_invalid_benchmark_path_contents(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(str(tmpdir / "test.bc"), "w") as f:
            f.write("Invalid bitcode")

        benchmark = Benchmark(
            uri="benchmark://new", program=File(uri=f"file:///{tmpdir}/test.bc")
        )

        with pytest.raises(ValueError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == 'Failed to parse LLVM bitcode: "benchmark://new"'


def test_benchmark_path_invalid_protocol(env: CompilerEnv):
    benchmark = Benchmark(
        uri="benchmark://new", program=File(uri="invalid_protocol://test")
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert (
        str(ctx.value)
        == 'Unsupported benchmark URI protocol: "invalid_protocol://test"'
    )


if __name__ == "__main__":
    main()
