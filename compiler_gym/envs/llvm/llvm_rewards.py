# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines reward spaces used by the LLVM environment."""
from typing import List, Optional

from compiler_gym.datasets import Benchmark
from compiler_gym.service import observation_t
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import RewardType
from compiler_gym.views.observation import ObservationView


class CostFunctionReward(Reward):
    """A reward function that uses a scalar observation space as a cost
    function.
    """

    __slots__ = [
        "cost_function",
        "init_cost_function",
        "previous_cost",
    ]

    def __init__(self, cost_function: str, init_cost_function: str, **kwargs):
        """Constructor.

        :param cost_function: The ID of the observation space used to produce
            scalar costs.
        :param init_cost_function: The ID of an observation space that produces
            a scalar cost equivalent to cost_function before any actions are
            made.
        """
        super().__init__(observation_spaces=[cost_function], **kwargs)
        self.cost_function: str = cost_function
        self.init_cost_function: str = init_cost_function
        self.previous_cost: Optional[observation_t] = None

    def reset(self, benchmark: Benchmark) -> None:
        """Called on env.reset(). Reset incremental progress."""
        del benchmark  # unused
        self.previous_cost = None

    def update(
        self,
        action: int,
        observations: List[observation_t],
        observation_view: ObservationView,
    ) -> RewardType:
        """Called on env.step(). Compute and return new reward."""
        cost: RewardType = observations[0]
        if self.previous_cost is None:
            self.previous_cost = observation_view[self.init_cost_function]
        reward = RewardType(self.previous_cost - cost)
        self.previous_cost = cost
        return reward


class NormalizedReward(CostFunctionReward):
    """A cost function reward that is normalized to the initial value."""

    __slots__ = ["cost_norm", "benchmark"]

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.cost_norm: Optional[observation_t] = None
        self.benchmark: Benchmark = None

    def reset(self, benchmark: str) -> None:
        """Called on env.reset(). Reset incremental progress."""
        super().reset(benchmark)
        # The benchmark has changed so we must compute a new cost normalization
        # value. If the benchmark has not changed then the previously computed
        # value is still valid.
        if self.benchmark != benchmark:
            self.cost_norm = None
            self.benchmark = benchmark

    def update(
        self,
        action: int,
        observations: List[observation_t],
        observation_view: ObservationView,
    ) -> RewardType:
        """Called on env.step(). Compute and return new reward."""
        if self.cost_norm is None:
            self.cost_norm = self.get_cost_norm(observation_view)
        return super().update(action, observations, observation_view) / self.cost_norm

    def get_cost_norm(self, observation_view: ObservationView) -> RewardType:
        """Return the value used to normalize costs."""
        return observation_view[self.init_cost_function]


class BaselineImprovementNormalizedReward(NormalizedReward):
    """A cost function reward that is normalized to improvement made by a
    baseline approach.
    """

    __slots__ = ["baseline_cost_function"]

    def __init__(self, baseline_cost_function: str, **kwargs):
        super().__init__(**kwargs)
        self.baseline_cost_function: str = baseline_cost_function

    def get_cost_norm(self, observation_view: ObservationView) -> RewardType:
        """Return the value used to normalize costs."""
        init_cost = observation_view[self.init_cost_function]
        baseline_cost = observation_view[self.baseline_cost_function]
        return max(init_cost - baseline_cost, 1)
