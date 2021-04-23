# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import CompilerEnv
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_add_benchmark_invalid_protocol(env: CompilerEnv):
    with pytest.raises(ValueError) as ctx:
        env.reset(
            benchmark=Benchmark(
                BenchmarkProto(
                    uri="benchmark://foo", program=File(uri="https://invalid/protocol")
                ),
            )
        )
    assert str(ctx.value) == (
        "Invalid benchmark data URI. "
        'Only the file:/// protocol is supported: "https://invalid/protocol"'
    )


def test_add_benchmark_invalid_path(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "not_a_file"
        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(benchmark=Benchmark.from_file("benchmark://foo", tmp))
        # Use endswith() because on macOS there may be a /private prefix.
        assert str(ctx.value).endswith(str(tmp))


if __name__ == "__main__":
    main()
