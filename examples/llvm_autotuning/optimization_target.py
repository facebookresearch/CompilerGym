# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Union

import numpy as np
from llvm_autotuning.just_keep_going_env import JustKeepGoingEnv

import compiler_gym
from compiler_gym.datasets import Benchmark
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import RuntimePointEstimateReward

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = Lock()


class OptimizationTarget(str, Enum):
    CODESIZE = "codesize"
    BINSIZE = "binsize"
    RUNTIME = "runtime"

    @property
    def optimization_space_enum_name(self) -> str:
        return {
            OptimizationTarget.CODESIZE: "IrInstructionCount",
            OptimizationTarget.BINSIZE: "ObjectTextSizeBytes",
            OptimizationTarget.RUNTIME: "Runtime",
        }[self.value]

    def make_env(self, benchmark: Union[str, Benchmark]) -> LlvmEnv:
        env: LlvmEnv = compiler_gym.make("llvm-v0")

        # TODO(cummins): This does not work with custom benchmarks, as the URI
        # will not be known to the new environment.
        if str(benchmark).startswith("file:///"):
            benchmark = env.make_benchmark(Path(benchmark[len("file:///") :]))

        env.benchmark = benchmark

        if self.value == OptimizationTarget.CODESIZE:
            env.reward_space = "IrInstructionCountOz"
        elif self.value == OptimizationTarget.BINSIZE:
            env.reward_space = "ObjectTextSizeOz"
        elif self.value == OptimizationTarget.RUNTIME:
            env = RuntimePointEstimateReward(env, warmup_count=0, runtime_count=3)
        else:
            assert False, f"Unknown OptimizationTarget: {self.value}"

        # Wrap the env to ignore errors during search.
        env = JustKeepGoingEnv(env)

        return env

    def final_reward(self, env: LlvmEnv, runtime_count: int = 30) -> float:
        """Compute the final reward of the environment.

        Note that this may modify the environment state. You should call
        :code:`reset()` before continuing to use the environment after this.
        """
        # Reapply the environment state in a retry loop.
        actions = list(env.actions)
        env.reset()
        for i in range(1, 5 + 1):
            _, _, done, info = env.step(actions)
            if not done:
                break
            logger.warning(
                "Attempt %d to apply actions during final reward failed: %s",
                i,
                info.get("error_details"),
            )
        else:
            raise ValueError("Failed to replay environment's actions")

        if self.value == OptimizationTarget.CODESIZE:
            return env.observation.IrInstructionCountOz() / max(
                env.observation.IrInstructionCount(), 1
            )

        if self.value == OptimizationTarget.BINSIZE:
            return env.observation.ObjectTextSizeOz() / max(
                env.observation.ObjectTextSizeBytes(), 1
            )

        if self.value == OptimizationTarget.RUNTIME:
            with _RUNTIME_LOCK:
                with compiler_gym.make("llvm-v0", benchmark=env.benchmark) as new_env:
                    new_env.reset()
                    new_env.runtime_observation_count = runtime_count
                    new_env.runtime_warmup_count = 0
                    new_env.apply(env.state)
                    final_runtimes = new_env.observation.Runtime()
                    assert len(final_runtimes) == runtime_count

                    new_env.reset()
                    new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
                    o3_runtimes = new_env.observation.Runtime()
                    assert len(o3_runtimes) == runtime_count

                logger.debug("O3 runtimes: %s", o3_runtimes)
                logger.debug("Final runtimes: %s", final_runtimes)
                speedup = np.median(o3_runtimes) / max(np.median(final_runtimes), 1e-12)
                logger.debug("Speedup: %.4f", speedup)

                return speedup

        assert False, f"Unknown OptimizationTarget: {self.value}"
