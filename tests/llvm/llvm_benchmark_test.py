# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import ClientServiceCompilerEnv
from compiler_gym.envs.llvm import llvm_benchmark
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from tests.pytest_plugins.common import macos_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_add_benchmark_invalid_scheme(env: ClientServiceCompilerEnv):
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


def test_add_benchmark_invalid_path(env: ClientServiceCompilerEnv):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "not_a_file"
        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(benchmark=Benchmark.from_file("benchmark://foo", tmp))
        # Use endswith() because on macOS there may be a /private prefix.
        assert str(ctx.value).endswith(str(tmp))


def test_get_system_library_flags_not_found():
    with pytest.raises(
        llvm_benchmark.HostCompilerFailure, match="Failed to invoke 'not-a-real-binary'"
    ):
        llvm_benchmark.get_system_library_flags("not-a-real-binary")


def test_get_system_library_flags_nonzero_exit_status():
    """Test that setting the $CXX to an invalid binary raises an error."""
    with pytest.raises(
        llvm_benchmark.HostCompilerFailure, match="Failed to invoke 'false'"
    ):
        llvm_benchmark.get_system_library_flags("false")


def test_get_system_library_flags_output_parse_failure():
    """Test that setting the $CXX to an invalid binary raises an error."""
    with pytest.raises(
        llvm_benchmark.UnableToParseHostCompilerOutput,
        match="Failed to parse '#include <...>' search paths from 'echo'",
    ):
        llvm_benchmark.get_system_library_flags("echo")


def test_get_system_library_flags():
    flags = llvm_benchmark.get_system_library_flags()
    assert flags
    assert "-isystem" in flags


@macos_only
def test_get_system_library_flags_system_libraries():
    flags = llvm_benchmark.get_system_library_flags()
    assert flags
    assert flags[-1] == "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"


def test_ClangInvocation_system_libs():
    cmd = llvm_benchmark.ClangInvocation(["foo.c"]).command("a.out")
    assert "-isystem" in cmd


def test_ClangInvocation_no_system_libs():
    cmd = llvm_benchmark.ClangInvocation(["foo.c"], system_includes=False).command(
        "a.out"
    )
    assert "-isystem" not in cmd


if __name__ == "__main__":
    main()
