# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from compiler_gym.errors import BenchmarkInitError, ServiceError
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType

import scipy
import numpy as np

class RuntimeSeriesReward(Reward):
    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        default_value: int = 0,
    ):
        super().__init__(
            name="runtime",
            observation_spaces=["Runtime"],
            default_value=default_value,
            min=None,
            max=None,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.runtime_count = runtime_count
        self.warmup_count = warmup_count
        self.starting_runtimes: List[float] = None
        self.previous_runtimes: List[float] = None
        self.current_benchmark: Optional[str] = None

    def reset(self, benchmark, observation_view) -> None:
        # If we are changing the benchmark then check that it is runnable.
        if benchmark != self.current_benchmark:
            if not observation_view["IsRunnable"]:
                raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")
            self.current_benchmark = benchmark
            self.starting_runtimes = None

        # Compute initial runtimes
        if self.starting_runtimes is None:
            self.starting_runtimes = observation_view["Runtime"]

        self.previous_runtimes = self.starting_runtimes

    def update(
        self,
        actions: List[ActionType],
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

        # Use the Kruskal–Wallis test to determine if the medians are equal
        # between the two series of runtimes. If the runtimes medians are
        # significantly different, compute the reward by computing the
        # difference between the two medians. Otherwise, set the reward as 0.
        # https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
        _, pval = scipy.stats.kruskal(runtimes, self.previous_runtimes)
        reward = np.median(self.previous_runtimes) - np.median(runtimes) if pval < 0.05 else 0
        self.previous_runtimes = runtimes
        return reward
