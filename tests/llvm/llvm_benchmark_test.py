# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import re
import tempfile
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.llvm import llvm_benchmark as llvm
from compiler_gym.errors.dataset_errors import BenchmarkInitError
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from tests.pytest_plugins.common import macos_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_add_benchmark_invalid_scheme(env: CompilerEnv):
    with pytest.raises(ValueError) as ctx:
        env.reset(
            benchmark=Benchmark(
                BenchmarkProto(
                    uri="benchmark://foo", program=File(uri="https://invalid/scheme")
                ),
            )
        )
    assert str(ctx.value) == (
        "Invalid benchmark data URI. "
        'Only the file:/// scheme is supported: "https://invalid/scheme"'
    )


def test_add_benchmark_invalid_path(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "not_a_file"
        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(benchmark=Benchmark.from_file("benchmark://foo", tmp))
        # Use endswith() because on macOS there may be a /private prefix.
        assert str(ctx.value).endswith(str(tmp))


def test_get_system_library_flags_not_found():
    with pytest.raises(
        llvm.HostCompilerFailure, match="Failed to invoke 'not-a-real-binary'"
    ):
        llvm.get_system_library_flags("not-a-real-binary")


def test_get_system_library_flags_nonzero_exit_status():
    """Test that setting the $CXX to an invalid binary raises an error."""
    with pytest.raises(llvm.HostCompilerFailure, match="Failed to invoke 'false'"):
        llvm.get_system_library_flags("false")


def test_get_system_library_flags_output_parse_failure():
    """Test that setting the $CXX to an invalid binary raises an error."""
    with pytest.raises(
        llvm.UnableToParseHostCompilerOutput,
        match="Failed to parse '#include <...>' search paths from 'echo'",
    ):
        llvm.get_system_library_flags("echo")


def test_get_system_library_flags():
    flags = llvm.get_system_library_flags()
    assert flags
    assert "-isystem" in flags


@macos_only
def test_get_system_library_flags_system_libraries():
    flags = llvm.get_system_library_flags()
    assert flags
    assert flags[-1] == "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"


def test_ClangInvocation_system_libs():
    cmd = llvm.ClangInvocation(["foo.c"]).command("a.out")
    assert "-isystem" in cmd


def test_ClangInvocation_no_system_libs():
    cmd = llvm.ClangInvocation(["foo.c"], system_includes=False).command("a.out")
    assert "-isystem" not in cmd


@pytest.mark.parametrize(
    "source",
    [
        "",
        "int A() {return 0;}",
        """
int A() {return 0;}
int B() {return A();}
int C() {return 0;}
    """,
    ],
)
@pytest.mark.parametrize("system_includes", [False, True])
def test_make_benchmark_from_source_valid_source(
    env: CompilerEnv, source: str, system_includes: bool
):
    benchmark = llvm.make_benchmark_from_source(source, system_includes=system_includes)
    env.reset(benchmark=benchmark)


@pytest.mark.parametrize(
    "source",
    [
        "@syntax error!!!",  # invalid syntax
        "int A() {return a;}",  # undefined variable
        '#include "missing.h"',  # missing include
    ],
)
@pytest.mark.parametrize("system_includes", [False, True])
def test_make_benchmark_from_source_invalid_source(source: str, system_includes: bool):
    with pytest.raises(
        BenchmarkInitError, match="Failed to make benchmark with compiler error:"
    ):
        llvm.make_benchmark_from_source(source, system_includes=system_includes)


def test_make_benchmark_from_source_invalid_copt():
    with pytest.raises(
        BenchmarkInitError, match="Failed to make benchmark with compiler error:"
    ):
        llvm.make_benchmark_from_source(
            "int A() {return 0;}", copt=["-invalid-argument!"]
        )


def test_make_benchmark_from_source_missing_system_includes():
    with pytest.raises(
        BenchmarkInitError, match="Failed to make benchmark with compiler error:"
    ):
        llvm.make_benchmark_from_source("#include <stdio.h>", system_includes=False)


def test_make_benchmark_from_source_with_system_includes():
    assert llvm.make_benchmark_from_source("#include <stdio.h>", system_includes=True)


def test_split_benchmark_by_function_no_functions():
    benchmark = llvm.make_benchmark_from_source("")
    with pytest.raises(ValueError, match="No functions found"):
        llvm.split_benchmark_by_function(benchmark)


def is_defined(signature: str, ir: str):
    """Return whether the function signature is defined in the IR."""
    return re.search(f"^define .*{signature}", ir, re.MULTILINE)


def is_declared(signature: str, ir: str):
    """Return whether the function signature is defined in the IR."""
    return re.search(f"^declare .*{signature}", ir, re.MULTILINE)


def test_split_benchmark_by_function_repeated_split_single_function(env: CompilerEnv):
    benchmark = llvm.make_benchmark_from_source("int A() {return 0;}", lang="c")
    for _ in range(10):
        benchmarks = llvm.split_benchmark_by_function(benchmark)
        assert len(benchmarks) == 1
        env.reset(benchmark=benchmarks[0])
        assert is_defined("i32 @A()", env.ir)
        benchmark = benchmarks[0]


def test_split_benchmark_by_function_multiple_functions(env: CompilerEnv):
    benchmark = llvm.make_benchmark_from_source(
        """
int A() {return 0;}
int B() {return A();}
""",
        lang="c",
    )

    benchmarks = llvm.split_benchmark_by_function(benchmark)
    assert len(benchmarks) == 2
    A, B = benchmarks

    env.reset(benchmark=A)
    assert is_defined("i32 @A()", env.ir)
    assert not is_defined("i32 @B()", env.ir)

    assert not is_declared("i32 @A()", env.ir)
    assert not is_declared("i32 @B()", env.ir)

    env.reset(benchmark=B)
    assert not is_defined("i32 @A()", env.ir)
    assert is_defined("i32 @B()", env.ir)

    assert is_declared("i32 @A()", env.ir)
    assert not is_declared("i32 @B()", env.ir)


def test_split_benchmark_by_function_maximum_function_count(env: CompilerEnv):
    benchmark = llvm.make_benchmark_from_source(
        """
int A() {return 0;}
int B() {return A();}
""",
        lang="c",
    )

    benchmarks = llvm.split_benchmark_by_function(
        benchmark,
        maximum_function_count=1,
    )
    assert len(benchmarks) == 1

    env.reset(benchmark=benchmarks[0])
    assert is_defined("i32 @A()", env.ir)


def test_merge_benchmarks_single_input(env: CompilerEnv):
    A = llvm.make_benchmark_from_source("int A() {return 0;}", lang="c")

    merged = llvm.merge_benchmarks([A])
    env.reset(benchmark=merged)

    assert is_defined("i32 @A()", env.ir)


def test_merge_benchmarks_independent(env: CompilerEnv):
    A = llvm.make_benchmark_from_source("int A() {return 0;}", lang="c")
    B = llvm.make_benchmark_from_source("int B() {return 0;}", lang="c")

    merged = llvm.merge_benchmarks([A, B])
    env.reset(benchmark=merged)

    assert is_defined("i32 @A()", env.ir)
    assert is_defined("i32 @B()", env.ir)


def test_merge_benchmarks_multiply_defined():
    A = llvm.make_benchmark_from_source("int A() {return 0;}", lang="c")
    with pytest.raises(ValueError, match="symbol multiply defined"):
        llvm.merge_benchmarks([A, A])


def test_merge_benchmarks_declarations(env: CompilerEnv):
    A = llvm.make_benchmark_from_source("int A() {return 0;}", lang="c")
    B = llvm.make_benchmark_from_source("int A(); int B() {return A();}", lang="c")

    merged = llvm.merge_benchmarks([A, B])
    env.reset(benchmark=merged)

    assert is_defined("i32 @A()", env.ir)
    assert is_defined("i32 @B()", env.ir)


if __name__ == "__main__":
    main()
