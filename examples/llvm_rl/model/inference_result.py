# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List

import numpy as np
from pydantic import BaseModel, validator
from ray.rllib.agents.dqn import ApexTrainer, R2D2Trainer  # noqa
from ray.rllib.agents.impala import ImpalaTrainer  # noqa
from ray.rllib.agents.ppo import PPOTrainer  # noqa

from compiler_gym.datasets import BenchmarkUri
from compiler_gym.envs import CompilerEnv
from compiler_gym.util.timer import Timer

logger = logging.getLogger(__name__)


class InferenceResult(BaseModel):
    """Represents the result of running an RL agent on a problem."""

    # The benchmark URI.
    benchmark: str
    inference_walltime_seconds: float
    commandline: str
    episode_len: int
    instruction_count_init: int
    instruction_count_final: int
    instruction_count_oz: int
    instruction_count_reduction: float
    """The final instruction count, normalized to -Oz."""
    object_size_init: int
    object_size_final: int
    object_size_oz: int
    object_size_reduction: float
    """The final object size, normalized to -Oz."""
    runtimes_init: List[float]
    runtimes_final: List[float]
    runtimes_o3: List[float]
    runtime_reduction: float
    """The final runtime, normalized to -Oz."""

    @classmethod
    def from_agent(
        cls, env: CompilerEnv, agent, runtime: bool = True, runtimes_count: int = 30
    ):
        # We calculate our own reward at the end, no need for incremental
        # rewards during inference.
        env.reward_space = None

        # Run inference on the environment.
        observation, done = env.reset(), False
        with Timer() as inference_timer:
            while not done:
                action = agent.compute_action(observation)
                observation, _, done, _ = env.step(action)

        instruction_count_init = env.unwrapped.observation["IrInstructionCountO0"]
        instruction_count_final = env.unwrapped.observation["IrInstructionCount"]
        instruction_count_oz = env.unwrapped.observation["IrInstructionCountOz"]

        object_size_init = env.unwrapped.observation["ObjectTextSizeO0"]
        object_size_final = env.unwrapped.observation["ObjectTextSizeBytes"]
        object_size_oz = env.unwrapped.observation["ObjectTextSizeOz"]

        runtimes_init = []
        runtimes_o3 = []
        runtimes_final = []

        try:
            if runtime and env.unwrapped.observation["IsRunnable"]:
                env.send_param(
                    "llvm.set_runtimes_per_observation_count", str(runtimes_count)
                )
                env.unwrapped.observation["Runtime"]  # warmup
                runtimes_final = env.unwrapped.observation["Runtime"].tolist()
                assert (
                    len(runtimes_final) == runtimes_count
                ), f"{len(runtimes_final)} != {runtimes_count}"

                env.reset()
                env.send_param(
                    "llvm.set_runtimes_per_observation_count", str(runtimes_count)
                )
                env.unwrapped.observation["Runtime"]  # warmup
                runtimes_init = env.unwrapped.observation["Runtime"].tolist()
                assert (
                    len(runtimes_init) == runtimes_count
                ), f"{len(runtimes_init)} != {runtimes_count}"

                env.send_param("llvm.apply_baseline_optimizations", "-O3")
                env.unwrapped.observation["Runtime"]  # warmup
                runtimes_o3 = env.unwrapped.observation["Runtime"].tolist()
                assert (
                    len(runtimes_o3) == runtimes_count
                ), f"{len(runtimes_o3)} != {runtimes_count}"
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to compute runtime: %s", e)

        return cls(
            benchmark=env.benchmark.uri,
            inference_walltime_seconds=inference_timer.time,
            commandline=env.commandline(),
            episode_len=len(env.actions),
            instruction_count_init=instruction_count_init,
            instruction_count_final=instruction_count_final,
            instruction_count_oz=instruction_count_oz,
            instruction_count_reduction=instruction_count_oz
            / max(instruction_count_final, 1),
            object_size_init=object_size_init,
            object_size_final=object_size_final,
            object_size_oz=object_size_oz,
            object_size_reduction=object_size_oz / max(object_size_final, 1),
            runtimes_init=runtimes_init,
            runtimes_final=runtimes_final,
            runtimes_o3=runtimes_o3,
            runtime_reduction=np.median(runtimes_o3 or [0])
            / max(np.median(runtimes_final or [0]), 1),
        )

    @validator("benchmark", pre=True)
    def validate_benchmark(cls, value):
        if isinstance(value, BenchmarkUri):
            return str(value)
        return value
