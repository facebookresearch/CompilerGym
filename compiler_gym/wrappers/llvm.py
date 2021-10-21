# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Wrapper classes for the LLVM environments."""
from typing import Callable, Iterable, List, Optional

import numpy as np

from compiler_gym.datasets.benchmark import BenchmarkInitError
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service.connection import ServiceError
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ObservationType
from compiler_gym.wrappers import CompilerEnvWrapper


class RuntimePointEstimateReward(CompilerEnvWrapper):
    """LLVM wrapper that uses a point estimate of program runtime as reward.

    This class wraps an LLVM environment and registers a new runtime reward
    space. Runtime is estimated from one or more runtime measurements, after
    optionally running one or more warmup runs. At each step, reward is the
    change in runtime estimate from the runtime estimate at the previous step.
    """

    class RuntimeReward(Reward):
        def __init__(
            self,
            runtime_count: int,
            warmup_count: int,
            estimator: Callable[[Iterable[float]], float],
        ):
            super().__init__(
                id="runtime",
                observation_spaces=["Runtime"],
                default_value=0,
                min=None,
                max=None,
                default_negates_returns=True,
                deterministic=False,
                platform_dependent=True,
            )
            self.runtime_count = runtime_count
            self.warmup_count = warmup_count
            self.starting_runtime: Optional[float] = None
            self.previous_runtime: Optional[float] = None
            self.current_benchmark: Optional[str] = None
            self.estimator = estimator

        def reset(self, benchmark, observation_view) -> None:
            # If we are changing the benchmark then check that it is runnable.
            if benchmark != self.current_benchmark:
                if not observation_view["IsRunnable"]:
                    raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")
                self.current_benchmark = benchmark
                self.starting_runtime = None

            # Compute initial runtime if required, else use previously computed
            # value.
            if self.starting_runtime is None:
                self.starting_runtime = self.estimator(observation_view["Runtime"])

            self.previous_runtime = self.starting_runtime

        def update(
            self,
            actions: List[int],
            observations: List[ObservationType],
            observation_view,
        ) -> float:
            del actions  # unused
            del observation_view  # unused
            runtimes = observations[0]
            if len(runtimes) != self.runtime_count:
                raise ServiceError(
                    f"Expected {self.runtime_count} runtimes but received {len(runtimes)}"
                )
            runtime = self.estimator(runtimes)

            reward = self.previous_runtime - runtime
            self.previous_runtime = runtime
            return reward

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
            self.RuntimeReward(
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
