# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM benchmark handling."""
import re
import subprocess
import tempfile
from pathlib import Path

import gym
import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import LlvmEnv, llvm
from compiler_gym.errors import BenchmarkInitError
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from compiler_gym.third_party import llvm as llvm_paths
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.temporary_working_directory import temporary_working_directory
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
        "benchmark://test_invalid_benchmark_data", "Invalid bitcode".encode("utf-8")
    )

    with pytest.raises(
        ValueError,
        match='Failed to parse LLVM bitcode: "benchmark://test_invalid_benchmark_data"',
    ):
        env.reset(benchmark=benchmark)


def test_invalid_benchmark_missing_file(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://test_invalid_benchmark_missing_file",
        )
    )

    with pytest.raises(ValueError, match="No program set in Benchmark:"):
        env.reset(benchmark=benchmark)


def test_benchmark_path_empty_file(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.bc").touch()

        benchmark = Benchmark.from_file(
            "benchmark://test_benchmark_path_empty_file", tmpdir / "test.bc"
        )

        with pytest.raises(ValueError, match="Failed to parse LLVM bitcode"):
            env.reset(benchmark=benchmark)


def test_invalid_benchmark_path_contents(env: LlvmEnv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(str(tmpdir / "test.bc"), "w") as f:
            f.write("Invalid bitcode")

        benchmark = Benchmark.from_file(
            "benchmark://test_invalid_benchmark_path_contents", tmpdir / "test.bc"
        )

        with pytest.raises(ValueError, match="Failed to parse LLVM bitcode"):
            env.reset(benchmark=benchmark)


def test_benchmark_path_invalid_scheme(env: LlvmEnv):
    benchmark = Benchmark(
        BenchmarkProto(
            uri="benchmark://test_benchmark_path_invalid_scheme",
            program=File(uri="invalid_scheme://test"),
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
    benchmark = Benchmark.from_file(
        "benchmark://test_custom_benchmark", EXAMPLE_BITCODE_FILE
    )
    env.reset(benchmark=benchmark)
    assert env.benchmark == "benchmark://test_custom_benchmark"


def test_custom_benchmark_constructor():
    benchmark = Benchmark.from_file(
        "benchmark://test_custom_benchmark_constructor", EXAMPLE_BITCODE_FILE
    )
    with gym.make("llvm-v0", benchmark=benchmark) as env:
        env.reset()
        assert env.benchmark == "benchmark://test_custom_benchmark_constructor"


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
    assert str(benchmark.uri).startswith("benchmark://user-v0/")
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


def test_failing_build_cmd(env: LlvmEnv, tmpdir):
    """Test that reset() raises an error if build command fails."""
    (Path(tmpdir) / "program.c").touch()

    benchmark = env.make_benchmark(Path(tmpdir) / "program.c")

    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        ["$CC", "$IN", "-invalid-cc-argument"]
    )
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = 10

    with pytest.raises(
        BenchmarkInitError,
        match=r"clang: error: unknown argument: '-invalid-cc-argument'",
    ):
        env.reset(benchmark=benchmark)


def test_make_benchmark_from_command_line_empty_input(env: LlvmEnv):
    with pytest.raises(ValueError, match="Input command line is empty"):
        env.make_benchmark_from_command_line("")
    with pytest.raises(ValueError, match="Input command line is empty"):
        env.make_benchmark_from_command_line([])


@pytest.mark.parametrize("cmd", ["gcc", ["gcc"]])
def test_make_benchmark_from_command_line_insufficient_args(env: LlvmEnv, cmd):
    with pytest.raises(ValueError, match="Input command line 'gcc' is too short"):
        env.make_benchmark_from_command_line(cmd)


@pytest.mark.parametrize("cmd", ["gcc in.c -o foo", ["gcc", "in.c", "-o", "foo"]])
def test_make_benchmark_from_command_line_build_cmd(env: LlvmEnv, cmd):
    with temporary_working_directory() as cwd:
        with open("in.c", "w") as f:
            f.write("int main() { return 0; }")

        bm = env.make_benchmark_from_command_line(cmd, system_includes=False)

        assert bm.proto.dynamic_config.build_cmd.argument[:4] == [
            str(llvm_paths.clang_path()),
            "-xir",
            "$IN",
            "-o",
        ]
        assert bm.proto.dynamic_config.build_cmd.argument[-1].endswith(f"{cwd}/foo")


@pytest.mark.parametrize("cmd", ["gcc in.c -o foo", ["gcc", "in.c", "-o", "foo"]])
def test_make_benchmark_from_command_line(env: LlvmEnv, cmd):
    with temporary_working_directory() as cwd:
        with open("in.c", "w") as f:
            f.write("int main() { return 0; }")

        bm = env.make_benchmark_from_command_line(cmd)
        assert not (cwd / "foo").is_file()

        env.reset(benchmark=bm)
        assert "main()" in env.ir

        assert (cwd / "foo").is_file()

        (cwd / "foo").unlink()
        bm.compile(env)
        assert (cwd / "foo").is_file()


def test_make_benchmark_from_command_line_no_system_includes(env: LlvmEnv):
    with temporary_working_directory():
        with open("in.c", "w") as f:
            f.write(
                """
#include <stdio.h>
int main() { return 0; }
"""
            )
        with pytest.raises(BenchmarkInitError, match="stdio.h"):
            env.make_benchmark_from_command_line("gcc in.c", system_includes=False)


def test_make_benchmark_from_command_line_system_includes(env: LlvmEnv):
    with temporary_working_directory():
        with open("in.c", "w") as f:
            f.write(
                """
#include <stdio.h>
int main() { return 0; }
"""
            )
        env.make_benchmark_from_command_line("gcc in.c")


def test_make_benchmark_from_command_line_stdin(env: LlvmEnv):
    with pytest.raises(ValueError, match="Input command line reads from stdin"):
        env.make_benchmark_from_command_line(["gcc", "-xc", "-"])


@pytest.mark.parametrize("retcode", [1, 5])
def test_make_benchmark_from_command_line_multiple_input_sources(
    env: LlvmEnv, retcode: int
):
    """Test that command lines with multiple source files are linked together."""
    with temporary_working_directory() as cwd:
        with open("a.c", "w") as f:
            f.write("int main() { return B(); }")

        with open("b.c", "w") as f:
            f.write(f"int B() {{ return {retcode}; }}")

        bm = env.make_benchmark_from_command_line(["gcc", "a.c", "b.c", "-o", "foo"])
        assert not (cwd / "foo").is_file()

        env.reset(benchmark=bm)
        assert "main()" in env.ir

        bm.compile(env)
        assert (cwd / "foo").is_file()

        p = subprocess.Popen(["./foo"])
        p.communicate(timeout=60)
        assert p.returncode == retcode


@pytest.mark.parametrize("retcode", [1, 5])
def test_make_benchmark_from_command_line_mixed_source_and_object_files(
    env: LlvmEnv, retcode: int
):
    """Test a command line that contains both source files and precompiled
    object files. The object files should be filtered from compilation but
    used for the final link.
    """
    with temporary_working_directory():
        with open("a.c", "w") as f:
            f.write(
                """
#include "b.h"

int A() {
    return B();
}

int main() {
    return A();
}
"""
            )

        with open("b.c", "w") as f:
            f.write(f"int B() {{ return {retcode}; }}")

        with open("b.h", "w") as f:
            f.write("int B();")

        # Compile b.c to object file:
        subprocess.check_call([str(llvm_paths.clang_path()), "b.c", "-c"], timeout=60)
        assert (Path("b.o")).is_file()

        bm = env.make_benchmark_from_command_line(["gcc", "a.c", "b.o", "-o", "foo"])
        env.reset(benchmark=bm)

        bm.compile(env)
        assert Path("foo").is_file()

        p = subprocess.Popen(["./foo"])
        p.communicate(timeout=60)
        assert p.returncode == retcode


def test_make_benchmark_from_command_line_only_object_files(env: LlvmEnv):
    with temporary_working_directory():
        with open("a.c", "w") as f:
            f.write("int A() { return 5; }")

        # Compile b.c to object file:
        subprocess.check_call([str(llvm_paths.clang_path()), "a.c", "-c"], timeout=60)
        assert (Path("a.o")).is_file()

        with pytest.raises(
            ValueError, match="Input command line has no source file inputs"
        ):
            env.make_benchmark_from_command_line(["gcc", "a.o", "-c"])


if __name__ == "__main__":
    main()
