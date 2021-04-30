# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM benchmark handling."""
import os
import re
import tempfile
from pathlib import Path

import gym
import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import LlvmEnv, llvm
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from compiler_gym.util.runfiles_path import runfiles_path
from tests.pytest_plugins.common import bazel_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# The path of an IR file that assembles but does not compile.
INVALID_IR_PATH = runfiles_path("tests/llvm/invalid_ir.ll")
EXAMPLE_BITCODE_FILE = runfiles_path(
    "compiler_gym/third_party/cbench/cbench-v1/crc32.bc"
)
EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT = 242


def test_reset_invalid_benchmark(env: LlvmEnv):
    invalid_benchmark = "an invalid benchmark"
    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=invalid_benchmark)

    assert str(ctx.value) == f"Invalid benchmark URI: 'benchmark://{invalid_benchmark}'"


def test_invalid_benchmark_data(env: LlvmEnv):
    benchmark = Benchmark.from_file_contents(
        "benchmark://new", "Invalid bitcode".encode("utf-8")
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == 'Failed to parse LLVM bitcode: "benchmark://new"'


def test_invalid_benchmark_missing_file(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://new",
        )
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == "No program set"


def test_benchmark_path_empty_file(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.bc").touch()

        benchmark = Benchmark.from_file("benchmark://new", tmpdir / "test.bc")

        with pytest.raises(ValueError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == f'File is empty: "{tmpdir}/test.bc"'


def test_invalid_benchmark_path_contents(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(str(tmpdir / "test.bc"), "w") as f:
            f.write("Invalid bitcode")

        benchmark = Benchmark.from_file("benchmark://new", tmpdir / "test.bc")

        with pytest.raises(ValueError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == 'Failed to parse LLVM bitcode: "benchmark://new"'


def test_benchmark_path_invalid_protocol(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://new", program=File(uri="invalid_protocol://test")
        ),
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert (
        str(ctx.value)
        == 'Invalid benchmark data URI. Only the file:/// protocol is supported: "invalid_protocol://test"'
    )


def test_custom_benchmark(env: LlvmEnv):
    benchmark = Benchmark.from_file("benchmark://new", EXAMPLE_BITCODE_FILE)
    env.reset(benchmark=benchmark)
    assert env.benchmark == "benchmark://new"


def test_custom_benchmark_constructor():
    benchmark = Benchmark.from_file("benchmark://new", EXAMPLE_BITCODE_FILE)
    env = gym.make("llvm-v0", benchmark=benchmark)
    try:
        env.reset()
        assert env.benchmark == "benchmark://new"
    finally:
        env.close()


def test_make_benchmark_single_bitcode(env: LlvmEnv):
    benchmark = llvm.make_benchmark(EXAMPLE_BITCODE_FILE)

    assert benchmark == f"file:///{EXAMPLE_BITCODE_FILE}"
    assert benchmark.proto.program.uri == f"file:///{EXAMPLE_BITCODE_FILE}"

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    assert env.observation["IrInstructionCount"] == EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT


@bazel_only
def test_make_benchmark_single_ll():
    """Test passing a single .ll file into make_benchmark()."""
    benchmark = llvm.make_benchmark(INVALID_IR_PATH)
    assert benchmark.uri.startswith("benchmark://user/")


def test_make_benchmark_single_clang_job(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "input.c"
        with open(str(source), "w") as f:
            f.write("int A() { return 0; }")

        benchmark = llvm.make_benchmark(str(source))

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    print(env.observation["Ir"])
    assert re.search(r"define (dso_local )?i32 @A\(\)", env.observation["Ir"])


def test_make_benchmark_split_clang_job(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as d:
        source_1 = Path(d) / "a.c"
        source_2 = Path(d) / "b.c"
        with open(str(source_1), "w") as f:
            f.write("int B() { return A(); }")
        with open(str(source_2), "w") as f:
            f.write("int A() { return 0; }")

        benchmark = llvm.make_benchmark(
            [
                str(source_1),
                str(source_2),
            ]
        )

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    print(env.observation["Ir"])
    assert re.search(r"define (dso_local )?i32 @A\(\)", env.observation["Ir"])
    assert re.search(r"define (dso_local )?i32 @B\(\)", env.observation["Ir"])


def test_make_benchmark_single_clang_invocation_multiple_inputs():
    with tempfile.TemporaryDirectory() as d:
        source_1 = Path(d) / "a.c"
        source_2 = Path(d) / "b.c"
        with open(str(source_1), "w") as f:
            f.write("int B() { return A(); }")
        with open(str(source_2), "w") as f:
            f.write("int A() { return 0; }")

        # cannot specify -o when generating multiple output files
        with pytest.raises(OSError):
            llvm.make_benchmark(llvm.ClangInvocation([str(source_1), str(source_2)]))


def test_make_benchmark_undefined_symbol(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "a.c"
        with open(str(source), "w") as f:
            f.write("int main() { return A(); }")

        benchmark = llvm.make_benchmark(source)

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    print(env.observation["Ir"])
    assert re.search(r"declare (dso_local )?i32 @A\(\.\.\.\)", env.observation["Ir"])


def test_make_benchmark_missing_file():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            llvm.make_benchmark(Path(d) / "a.c")

        with pytest.raises(FileNotFoundError):
            llvm.make_benchmark(str(Path(d) / "a.c"))


def test_make_benchmark_unrecognized_file_type():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "foo.txt"
        path.touch()

        with pytest.raises(ValueError) as ctx:
            llvm.make_benchmark(path)

        assert "Unrecognized file type" in str(ctx.value)


def test_make_benchmark_clang_job_standard_libraries(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "input.cc"
        with open(str(source), "w") as f:
            f.write('#include <stdio.h>\nint A() { printf(""); return 0; }')

        benchmark = llvm.make_benchmark(str(source))

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    print(env.observation["Ir"])
    assert re.search(r"define (dso_local )?i32 @_Z1Av\(\)", env.observation["Ir"])
    assert re.search(r"declare (dso_local )?i32 @printf", env.observation["Ir"])


def test_make_benchmark_invalid_clang_job():
    with pytest.raises(OSError) as ctx:
        llvm.make_benchmark(llvm.ClangInvocation(["-invalid-arg"]))

    assert "Compilation job failed with returncode" in str(ctx.value)
    assert "-invalid-arg" in str(ctx.value)


def test_custom_benchmark_is_added_on_service_restart(env: LlvmEnv):
    # When the service is restarted, the environment must send a custom
    # benchmark to it again.
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "a.c"
        with open(str(source), "w") as f:
            f.write("int main() { return 0; }")

        benchmark = llvm.make_benchmark(source)

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri

    # Kill the service so that the next call to reset() starts a new one.
    env.service.close()
    env.service = None

    env.reset()
    assert env.benchmark == benchmark.uri


def test_two_custom_benchmarks_reset(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "a.c"
        with open(str(source), "w") as f:
            f.write("int main() { return 0; }")

        benchmark1 = llvm.make_benchmark(source)
        benchmark2 = llvm.make_benchmark(source)

    assert benchmark1.uri != benchmark2.uri

    env.reset(benchmark=benchmark1)
    assert env.benchmark == benchmark1.uri
    env.reset()
    assert env.benchmark == benchmark1.uri
    with pytest.warns(
        UserWarning,
        match=r"Changing the benchmark has no effect until reset\(\) is called",
    ):
        env.benchmark = benchmark2
    env.reset()
    assert env.benchmark == benchmark2.uri


def test_get_system_includes_nonzero_exit_status():
    """Test that setting the $CXX to an invalid binary raises an error."""
    old_cxx = os.environ.get("CXX")
    os.environ["CXX"] = "false"
    try:
        with pytest.raises(OSError) as ctx:
            list(
                llvm.llvm_benchmark._get_system_includes()  # pylint: disable=protected-access
            )
        assert "Failed to invoke false" in str(ctx.value)
    finally:
        if old_cxx:
            os.environ["CXX"] = old_cxx


def test_get_system_includes_output_parse_failure():
    """Test that setting the $CXX to an invalid binary raises an error."""
    old_cxx = os.environ.get("CXX")
    os.environ["CXX"] = "echo"
    try:
        with pytest.raises(OSError) as ctx:
            list(
                llvm.llvm_benchmark._get_system_includes()  # pylint: disable=protected-access
            )
        assert "Failed to parse '#include <...>' search paths from echo" in str(
            ctx.value
        )
    finally:
        if old_cxx:
            os.environ["CXX"] = old_cxx


if __name__ == "__main__":
    main()
