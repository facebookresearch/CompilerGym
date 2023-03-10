# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for splitting and merging benchmarks."""
import random

import pytest

from compiler_gym.datasets import Benchmark
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm import llvm_benchmark as llvm
from compiler_gym.errors import BenchmarkInitError
from compiler_gym.validation_result import ValidationResult
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.mark.timeout(600)
def test_cbench_split_merge_build(env: LlvmEnv, validatable_cbench_uri: str):
    """Split and then merge a cBench program, checking that the merged program
    passes semantics validation.
    """
    env.reset(benchmark=validatable_cbench_uri, reward_space="IrInstructionCount")
    initial_instruction_count = env.observation.IrInstructionCount()

    split = llvm.split_benchmark_by_function(env.benchmark)
    merged = llvm.merge_benchmarks(split)

    # Copy over the dynamic configuration to enable runtime semantics
    # validation.
    merged.proto.dynamic_config.MergeFrom(env.benchmark.proto.dynamic_config)
    for cb in env.benchmark.validation_callbacks():
        merged.add_validation_callback(cb)

    env.reset(benchmark=merged)

    assert env.observation.IrInstructionCount() == initial_instruction_count

    result: ValidationResult = env.validate()
    assert not result.error_details
    assert result.reward_validated
    assert not result.actions_replay_failed
    assert not result.reward_validation_failed
    assert result.benchmark_semantics_validated
    assert not result.benchmark_semantics_validation_failed
    assert result.okay()


def test_cbench_split_globalopt_merge_safe_unsafe_actions(
    env: LlvmEnv, action_name: str
):
    """A test which shows that stripping symbols before split+merge causes
    invalid results.
    """
    safe = action_name not in {"-strip", "-strip-nondebug"}

    env.reset(benchmark="benchmark://cbench-v1/sha")
    env.step(env.action_space[action_name])
    ic = env.observation.IrInstructionCount()

    uri = f"benchmark://test-v0/{random.randrange(16**4):04x}"
    split = llvm.split_benchmark_by_function(
        Benchmark.from_file_contents(uri=uri, data=env.observation.Bitcode().tobytes())
    )

    def run_globalopt_on_benchmark(benchmark):
        env.reset(benchmark=benchmark)
        env.step(env.action_space["-globalopt"])
        return Benchmark.from_file_contents(
            uri=benchmark, data=env.observation.Bitcode().tobytes()
        )

    split = [run_globalopt_on_benchmark(s) for s in split]
    merged = llvm.merge_benchmarks(split)

    env.reset(benchmark=merged)

    if safe:
        assert env.observation.IrInstructionCount() == ic
    else:
        assert env.observation.IrInstructionCount() != ic


@pytest.mark.parametrize("action_name", ["-strip", "-strip-nondebug"])
def test_cbench_strip_unsafe_for_split(env: LlvmEnv, action_name: str):
    """Sanity check for test_cbench_split_globalopt_merge_safe_unsafe_actions()
    above. Run the two strip actions and show that they are safe to use if you
    don't split+merge.
    """
    env.reset(benchmark="benchmark://cbench-v1/sha")
    env.step(env.action_space[action_name])

    uri = f"benchmark://test-v0/{random.randrange(16**4):04x}"
    split = llvm.split_benchmark_by_function(
        Benchmark.from_file_contents(uri=uri, data=env.observation.Bitcode().tobytes())
    )
    merged = llvm.merge_benchmarks(split)

    # Copy over the dynamic config to compile the binary:
    merged.proto.dynamic_config.MergeFrom(env.benchmark.proto.dynamic_config)

    with pytest.raises(BenchmarkInitError):
        env.reset(benchmark=merged)


if __name__ == "__main__":
    main()
