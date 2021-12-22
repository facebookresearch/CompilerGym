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
    with pytest.raises(
        LookupError, match=f"Dataset not found: benchmark://{invalid_benchmark}"
    ):
        env.reset(benchmark=invalid_benchmark)


def test_invalid_benchmark_data(env: LlvmEnv):
    benchmark = Benchmark.from_file_contents(
        "benchmark://new", "Invalid bitcode".encode("utf-8")
    )

    with pytest.raises(
        ValueError, match='Failed to parse LLVM bitcode: "benchmark://new"'
    ):
        env.reset(benchmark=benchmark)


def test_invalid_benchmark_missing_file(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://new",
        )
    )

    with pytest.raises(ValueError, match="No program set"):
        env.reset(benchmark=benchmark)


def test_benchmark_path_empty_file(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.bc").touch()

        benchmark = Benchmark.from_file("benchmark://new", tmpdir / "test.bc")

        with pytest.raises(ValueError, match="Failed to parse LLVM bitcode"):
            env.reset(benchmark=benchmark)


def test_invalid_benchmark_path_contents(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(str(tmpdir / "test.bc"), "w") as f:
            f.write("Invalid bitcode")

        benchmark = Benchmark.from_file("benchmark://new", tmpdir / "test.bc")

        with pytest.raises(ValueError, match="Failed to parse LLVM bitcode"):
            env.reset(benchmark=benchmark)


def test_benchmark_path_invalid_scheme(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://new", program=File(uri="invalid_scheme://test")
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid benchmark data URI. "
            'Only the file:/// scheme is supported: "invalid_scheme://test"'
        ),
    ):
        env.reset(benchmark=benchmark)


def test_custom_benchmark(env: LlvmEnv):
    benchmark = Benchmark.from_file("benchmark://new", EXAMPLE_BITCODE_FILE)
    env.reset(benchmark=benchmark)
    assert env.benchmark == "benchmark://new"


def test_custom_benchmark_constructor():
    benchmark = Benchmark.from_file("benchmark://new", EXAMPLE_BITCODE_FILE)
    with gym.make("llvm-v0", benchmark=benchmark) as env:
        env.reset()
        assert env.benchmark == "benchmark://new"


def test_make_benchmark_single_bitcode(env: LlvmEnv):
    benchmark = llvm.make_benchmark(EXAMPLE_BITCODE_FILE)

    assert benchmark == f"benchmark://file-v0{EXAMPLE_BITCODE_FILE}"
    assert benchmark.uri.scheme == "benchmark"
    assert benchmark.uri.dataset == "file-v0"

    with open(EXAMPLE_BITCODE_FILE, "rb") as f:
        contents = f.read()

    assert benchmark.proto.program.contents == contents

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    assert env.observation["IrInstructionCount"] == EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT


@bazel_only
def test_make_benchmark_single_ll():
    """Test passing a single .ll file into make_benchmark()."""
    benchmark = llvm.make_benchmark(INVALID_IR_PATH)
    assert benchmark.uri.startswith("benchmark://user-v0/")
    assert benchmark.uri.scheme == "benchmark"
    assert benchmark.uri.dataset == "user-v0"


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

        with pytest.raises(ValueError, match=r"Unrecognized file type"):
            llvm.make_benchmark(path)


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
    with pytest.raises(OSError, match="Compilation job failed with returncode"):
        llvm.make_benchmark(llvm.ClangInvocation(["-invalid-arg"]))


def test_custom_benchmark_is_added_on_service_restart(env: LlvmEnv):
    # When the service is restarted, the environment still uses the same custom
    # benchmark.
    with tempfile.TemporaryDirectory() as d:
        source = Path(d) / "a.c"
        with open(str(source), "w") as f:
            f.write("int main() { return 0; }")

        benchmark = llvm.make_benchmark(source)

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri

    # Kill the service so that the next call to reset() starts a new one.
    env.close()
    assert env.service is None

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


def test_get_compiler_includes_not_found():
    with pytest.raises(OSError, match=r"Failed to invoke not-a-real-binary"):
        list(llvm.llvm_benchmark.get_compiler_includes("not-a-real-binary"))


def test_get_compiler_includes_nonzero_exit_status():
    """Test that setting the $CXX to an invalid binary raises an error."""
    with pytest.raises(OSError, match=r"Failed to invoke false"):
        list(llvm.llvm_benchmark.get_compiler_includes("false"))


def test_get_compiler_includes_output_parse_failure():
    """Test that setting the $CXX to an invalid binary raises an error."""
    old_cxx = os.environ.get("CXX")
    os.environ["CXX"] = "echo"
    try:
        with pytest.raises(
            OSError, match="Failed to parse '#include <...>' search paths from echo"
        ):
            list(llvm.llvm_benchmark.get_compiler_includes("echo"))
    finally:
        if old_cxx:
            os.environ["CXX"] = old_cxx


if __name__ == "__main__":
    main()
