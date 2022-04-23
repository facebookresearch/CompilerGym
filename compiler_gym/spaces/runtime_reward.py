# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterable, List, Optional

from compiler_gym.errors import BenchmarkInitError, ServiceError
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType


class RuntimeReward(Reward):
    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        estimator: Callable[[Iterable[float]], float],
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
        runtime = self.estimator(runtimes)

        reward = self.previous_runtime - runtime
        self.previous_runtime = runtime
        return reward
