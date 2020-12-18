# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import tempfile
from pathlib import Path

import pytest

from compiler_gym.envs import CompilerEnv
from compiler_gym.service.proto import Benchmark, File
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


def test_add_benchmark_invalid_protocol(env: CompilerEnv):
    with pytest.raises(ValueError) as ctx:
        env.reset(
            benchmark=Benchmark(
                uri="benchmark://foo", program=File(uri="https://invalid/protocol")
            )
        )
    assert (
        str(ctx.value)
        == 'Unsupported benchmark URI protocol: "https://invalid/protocol"'
    )


def test_add_benchmark_invalid_path(env: CompilerEnv):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "not_a_file"
        with pytest.raises(FileNotFoundError) as ctx:
            env.reset(
                benchmark=Benchmark(
                    uri="benchmark://foo", program=File(uri=f"file:///{tmp}")
                )
            )
        assert str(ctx.value) == f'File not found: "{tmp}"'


if __name__ == "__main__":
    main()
