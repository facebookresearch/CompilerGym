# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for LLVM runtime support."""
from pathlib import Path

import numpy as np
import pytest
from flaky import flaky

from compiler_gym.datasets import BenchmarkInitError
from compiler_gym.envs.llvm import LlvmEnv, llvm_benchmark
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.mark.parametrize("runtime_observation_count", [1, 3, 5])
def test_custom_benchmark_runtime(env: LlvmEnv, tmpdir, runtime_observation_count: int):
    env.reset()
    env.runtime_observation_count = runtime_observation_count
    with open(tmpdir / "program.c", "w") as f:
        f.write(
            """
    #include <stdio.h>

    int main(int argc, char** argv) {
        printf("Hello\\n");
        for (int i = 0; i < 10; ++i) {
            argc += 2;
        }
        return argc - argc;
    }
        """
        )

    benchmark = env.make_benchmark(Path(tmpdir) / "program.c")

    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        ["$CC", "$IN"] + llvm_benchmark.get_system_library_flags()
    )
    benchmark.proto.dynamic_config.build_cmd.outfile.extend(["a.out"])
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = 10

    benchmark.proto.dynamic_config.run_cmd.argument.extend(["./a.out"])
    benchmark.proto.dynamic_config.run_cmd.timeout_seconds = 10

    env.reset(benchmark=benchmark)
    runtimes = env.observation.Runtime()
    assert len(runtimes) == runtime_observation_count
    assert np.all(runtimes > 0)


@flaky
def test_custom_benchmark_runtimes_differ(env: LlvmEnv, tmpdir):
    """Same as above, but test that runtimes differ from run to run."""
    env.reset()

    env.runtime_observation_count = 10
    with open(tmpdir / "program.c", "w") as f:
        f.write(
            """
    #include <stdio.h>

    int main(int argc, char** argv) {
        printf("Hello\\n");
        for (int i = 0; i < 10; ++i) {
            argc += 2;
        }
        return argc - argc;
    }
        """
        )

    benchmark = env.make_benchmark(Path(tmpdir) / "program.c")

    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        ["$CC", "$IN"] + llvm_benchmark.get_system_library_flags()
    )
    benchmark.proto.dynamic_config.build_cmd.outfile.extend(["a.out"])
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = 10

    benchmark.proto.dynamic_config.run_cmd.argument.extend(["./a.out"])
    benchmark.proto.dynamic_config.run_cmd.timeout_seconds = 10

    env.reset(benchmark=benchmark)
    runtimes_a = env.observation.Runtime()
    runtimes_b = env.observation.Runtime()
    assert not np.all(runtimes_a == runtimes_b)


def test_failing_build_cmd(env: LlvmEnv, tmpdir):
    """Test that Runtime observation raises an error if build command fails."""
    with open(tmpdir / "program.c", "w") as f:
        f.write(
            """
    #include <stdio.h>

    int main(int argc, char** argv) {
        printf("Hello\\n");
        for (int i = 0; i < 10; ++i) {
            argc += 2;
        }
        return argc - argc;
    }
        """
        )

    benchmark = env.make_benchmark(Path(tmpdir) / "program.c")

    benchmark.proto.dynamic_config.build_cmd.argument.extend(
        ["$CC", "$IN", "-invalid-cc-argument"]
    )
    benchmark.proto.dynamic_config.build_cmd.outfile.extend(["a.out"])
    benchmark.proto.dynamic_config.build_cmd.timeout_seconds = 10

    benchmark.proto.dynamic_config.run_cmd.argument.extend(["./a.out"])
    benchmark.proto.dynamic_config.run_cmd.timeout_seconds = 10

    env.reset(benchmark=benchmark)
    with pytest.raises(
        BenchmarkInitError,
        match="clang: error: unknown argument: '-invalid-cc-argument'",
    ):
        env.observation.Runtime()


def test_invalid_runtime_count(env: LlvmEnv):
    env.reset()
    with pytest.raises(
        ValueError, match=r"runtimes_per_observation_count must be >= 1"
    ):
        env.runtime_observation_count = 0

    with pytest.raises(
        ValueError, match=r"runtimes_per_observation_count must be >= 1"
    ):
        env.runtime_observation_count = -1


def test_runtime_observation_count_before_reset(env: LlvmEnv):
    """Test setting property before reset() is called."""
    env.runtime_observation_count = 10
    assert env.runtime_observation_count == 10
    env.reset()
    assert env.runtime_observation_count == 10


def test_runtime_warmup_runs_count_before_reset(env: LlvmEnv):
    """Test setting property before reset() is called."""
    env.runtime_warmup_runs_count = 10
    assert env.runtime_warmup_runs_count == 10
    env.reset()
    assert env.runtime_warmup_runs_count == 10


def test_runtime_observation_count_fork(env: LlvmEnv):
    """Test that custom count properties propagate on fork()."""
    env.runtime_observation_count = 2
    env.runtime_warmup_runs_count = 1

    with env.fork() as fkd:
        assert fkd.runtime_observation_count == 2
        assert fkd.runtime_warmup_runs_count == 1

    env.reset()
    with env.fork() as fkd:
        assert fkd.runtime_observation_count == 2
        assert fkd.runtime_warmup_runs_count == 1


def test_default_runtime_observation_count_fork(env: LlvmEnv):
    """Test that default property values propagate on fork()."""
    env.reset()
    rc = env.runtime_observation_count
    wc = env.runtime_warmup_runs_count

    with env.fork() as fkd:
        assert fkd.runtime_observation_count == rc
        assert fkd.runtime_warmup_runs_count == wc


if __name__ == "__main__":
    main()
