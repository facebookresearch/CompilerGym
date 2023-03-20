# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Wrapper classes for the LLVM environments."""
from typing import Callable, Iterable

import numpy as np

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.spaces import RuntimeReward
from compiler_gym.spaces import RuntimeSeriesReward
from compiler_gym.wrappers import CompilerEnvWrapper


class RuntimePointEstimateReward(CompilerEnvWrapper):
    """LLVM wrapper that uses a point estimate of program runtime as reward.

    This class wraps an LLVM environment and registers a new runtime reward
    space. Runtime is estimated from one or more runtime measurements, after
    optionally running one or more warmup runs. At each step, reward is the
    change in runtime estimate from the runtime estimate at the previous step.
    """

    def __init__(
        self,
        env: LlvmEnv,
        runtime_count: int = 30,
        warmup_count: int = 0,
        estimator: Callable[[Iterable[float]], float] = np.median,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param runtime_count: The number of times to execute the binary when
            estimating the runtime.

        :param warmup_count: The number of warmup runs of the binary to perform
            before measuring the runtime.

        :param estimator: A function that takes a list of runtime measurements
            and produces a point estimate.
        """
        super().__init__(env)

        self.env.unwrapped.reward.add_space(
            RuntimeReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
                estimator=estimator,
            )
        )
        self.env.unwrapped.reward_space = "runtime"

        self.env.unwrapped.runtime_observation_count = runtime_count
        self.env.unwrapped.runtime_warmup_runs_count = warmup_count

    def fork(self) -> "RuntimePointEstimateReward":
        fkd = self.env.fork()
        # Remove the original "runtime" space so that we that new
        # RuntimePointEstimateReward wrapper instance does not attempt to
        # redefine, raising a warning.
        del fkd.unwrapped.reward.spaces["runtime"]
        return RuntimePointEstimateReward(
            env=fkd,
            runtime_count=self.reward.spaces["runtime"].runtime_count,
            warmup_count=self.reward.spaces["runtime"].warmup_count,
            estimator=self.reward.spaces["runtime"].estimator,
        )

class RuntimeSeriesEstimateReward(CompilerEnvWrapper):
    """LLVM wrapper that estimates the runtime of a program using N runtime
    observations and uses it as the reward.

    This class wraps an LLVM environment and registers a new runtime reward
    space. It is similar to the RuntimePointEstimateReward except that it only
    computes runtime differences if the change in runtime is significantly
    different from the runtimes in the previous step.

    See RuntimeSeriesReward for more details.
    """

    def __init__(
        self,
        env: LlvmEnv,
        runtime_count: int = 30,
        warmup_count: int = 0,
    ):
        """Constructor.
        :param env: The environment to wrap.
        :param runtime_count: The number of times to execute the binary when
            estimating the runtime.
        :param warmup_count: The number of warmup runs of the binary to perform
            before measuring the runtime.
        """
        super().__init__(env)

        self.env.unwrapped.reward.add_space(
            RuntimeSeriesReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
            )
        )
        self.env.unwrapped.reward_space = "runtimeseries"

        self.env.unwrapped.runtime_observation_count = runtime_count
        self.env.unwrapped.runtime_warmup_runs_count = warmup_count

    def fork(self) -> "RuntimeSeriesEstimateReward":
        fkd = self.env.fork()
        # Remove the original "runtimeseries" space so that we that new
        # RuntimeSeriesEstimateReward wrapper instance does not attempt to
        # redefine, raising a warning.
        del fkd.unwrapped.reward.spaces["runtimeseries"]
        return RuntimeSeriesEstimateReward(
            env=fkd,
            runtime_count=self.reward.spaces["runtimeseries"].runtime_count,
            warmup_count=self.reward.spaces["runtimeseries"].warmup_count,
        )
