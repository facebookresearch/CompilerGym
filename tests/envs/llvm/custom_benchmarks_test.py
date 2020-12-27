# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM benchmark handling."""
import os
import re
import tempfile
from pathlib import Path

import pytest

from compiler_gym.envs import LlvmEnv, llvm
from compiler_gym.service.proto import Benchmark, File
from compiler_gym.util.runfiles_path import runfiles_path
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

EXAMPLE_BITCODE_FILE = runfiles_path(
    "CompilerGym/compiler_gym/third_party/cBench/cBench/crc32.bc"
)
EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT = 196


def test_reset_invalid_benchmark(env: LlvmEnv):
    invalid_benchmark = "an invalid benchmark"
    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=invalid_benchmark)

    assert str(ctx.value) == f'Unknown benchmark "{invalid_benchmark}"'


def test_invalid_benchmark_data(env: LlvmEnv):
    benchmark = Benchmark(
        uri="benchmark://new", program=File(contents="Invalid bitcode".encode("utf-8"))
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == 'Failed to parse LLVM bitcode: "benchmark://new"'


def test_invalid_benchmark_missing_file(env: LlvmEnv):
    benchmark = Benchmark(
        uri="benchmark://new",
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert str(ctx.value) == "No program set"


def test_benchmark_path_not_found(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        benchmark = Benchmark(
            uri="benchmark://new", program=File(uri=f"file:///{tmpdir}/not_found")
        )

        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == f'File not found: "{tmpdir}/not_found"'


def test_benchmark_path_empty_file(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.bc").touch()

        benchmark = Benchmark(
            uri="benchmark://new", program=File(uri=f"file:///{tmpdir}/test.bc")
        )

        with pytest.raises(ValueError) as ctx:
            env.reset(benchmark=benchmark)

    assert str(ctx.value) == f'File is empty: "{tmpdir}/test.bc"'


def test_invalid_benchmark_path_contents(env: LlvmEnv):
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


def test_benchmark_path_invalid_protocol(env: LlvmEnv):
    benchmark = Benchmark(
        uri="benchmark://new", program=File(uri="invalid_protocol://test")
    )

    with pytest.raises(ValueError) as ctx:
        env.reset(benchmark=benchmark)

    assert (
        str(ctx.value)
        == 'Unsupported benchmark URI protocol: "invalid_protocol://test"'
    )


def test_custom_benchmark(env: LlvmEnv):
    benchmark = Benchmark(
        uri="benchmark://new", program=File(uri=f"file:///{EXAMPLE_BITCODE_FILE}")
    )
    env.reset(benchmark=benchmark)
    assert env.benchmark == "benchmark://new"


def test_make_benchmark_single_bitcode(env: LlvmEnv):
    benchmark = llvm.make_benchmark(EXAMPLE_BITCODE_FILE)

    assert benchmark.uri == f"file:///{EXAMPLE_BITCODE_FILE}"
    assert benchmark.program.uri == f"file:///{EXAMPLE_BITCODE_FILE}"

    env.reset(benchmark=benchmark)
    assert env.benchmark == benchmark.uri
    assert env.observation["IrInstructionCount"] == EXAMPLE_BITCODE_IR_INSTRUCTION_COUNT


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
    env.benchmark = benchmark2
    # assert env.benchmark == benchmark1.uri
    env.reset()
    assert env.benchmark == benchmark2.uri


def test_get_system_includes_nonzero_exit_status():
    """Test that setting the $CXX to an invalid binary raises an error."""
    old_cxx = os.environ.get("CXX")
    os.environ["CXX"] = "false"
    try:
        with pytest.raises(OSError) as ctx:
            list(llvm.benchmarks._get_system_includes())
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
            list(llvm.benchmarks._get_system_includes())
        assert "Failed to parse '#include <...>' search paths from echo" in str(
            ctx.value
        )
    finally:
        if old_cxx:
            os.environ["CXX"] = old_cxx


if __name__ == "__main__":
    main()
