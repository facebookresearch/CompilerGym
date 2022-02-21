# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import os
import tempfile
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.envs.llvm import llvm_benchmark
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


def test_get_system_library_flags_not_found(caplog):
    flags, error = llvm_benchmark._get_cached_system_library_flags("not-a-real-binary")
    assert flags == []
    assert "Failed to invoke not-a-real-binary" in error


def test_get_system_library_flags_nonzero_exit_status(caplog):
    """Test that setting the $CXX to an invalid binary raises an error."""
    flags, error = llvm_benchmark._get_cached_system_library_flags("false")
    assert flags == []
    assert "Failed to invoke false" in error


def test_get_system_library_flags_output_parse_failure(caplog):
    """Test that setting the $CXX to an invalid binary raises an error."""
    old_cxx = os.environ.get("CXX")
    try:
        os.environ["CXX"] = "echo"
        flags, error = llvm_benchmark._get_cached_system_library_flags("echo")
        assert flags == []
        assert "Failed to parse '#include <...>' search paths from echo" in error
    finally:
        if old_cxx:
            os.environ["CXX"] = old_cxx


def test_get_system_library_flags():
    flags = llvm_benchmark.get_system_library_flags()
    assert flags
    assert "-isystem" in flags


@macos_only
def test_get_system_library_flags_system_libraries():
    flags = llvm_benchmark.get_system_library_flags()
    assert flags
    assert flags[-1] == "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"


if __name__ == "__main__":
    main()
