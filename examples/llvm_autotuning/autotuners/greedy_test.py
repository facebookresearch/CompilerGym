# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for the LLVM autotuners."""
import pytest
from llvm_autotuning.autotuners import Autotuner

import compiler_gym


@pytest.mark.skip(reason="greedy takes a long time")
def test_autotune():
    with compiler_gym.make("llvm-v0", reward_space="IrInstructionCount") as env:
        env.reset(benchmark="benchmark://cbench-v1/crc32")

        autotuner = Autotuner(
            algorithm="greedy",
            optimization_target="codesize",
            search_time_seconds=3,
        )

        result = autotuner(env)
        print(result)
        assert result.benchmark == "benchmark://cbench-v1/crc32"
        assert result.walltime >= 3
        assert result.commandline == env.commandline()
        assert env.episode_reward
        assert env.benchmark == "benchmark://cbench-v1/crc32"
        assert env.reward_space == "IrInstructionCount"
