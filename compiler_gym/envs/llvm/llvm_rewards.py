# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This file defines reward spaces used by the LLVM environment."""
from typing import List, Optional

from compiler_gym.service import observation_t
from compiler_gym.spaces.reward import Reward
from compiler_gym.views.observation import ObservationView


class CostFunctionReward(Reward):
    """A reward function that uses an observation as a cost function."""

    __slots__ = [
        "cost_function",
        "init_cost_function",
        "forced_init_cost",
        "previous_cost",
    ]

    def __init__(
        self,
        cost_function: str,
        init_cost_function: str,
        forced_init_cost: Optional[observation_t] = None,
        **kwargs
    ):
        super().__init__(observation_spaces=[cost_function], **kwargs)
        self.cost_function: str = cost_function
        self.init_cost_function: str = init_cost_function
        self.forced_init_cost: Optional[observation_t] = forced_init_cost
        self.previous_cost: Optional[observation_t] = None

    def reset(self) -> None:
        self.previous_cost = None

    def update(
        self,
        action: int,
        observations: List[observation_t],
        observation_view: ObservationView,
    ) -> float:
        cost: float = observations[0]
        if self.previous_cost is None:
            if self.forced_init_cost is None:
                self.previous_cost = observation_view[self.init_cost_function]
            else:
                self.previous_cost = self.forced_init_cost
        reward = float(self.previous_cost - cost)
        self.previous_cost = cost
        return reward


class NormalizedReward(CostFunctionReward):
    """A cost function reward that is normalized to an initial value."""

    __slots__ = ["forced_cost_norm", "cost_norm"]

    def __init__(self, forced_cost_norm: Optional[observation_t] = None, **kwargs):
        super().__init__(**kwargs)
        self.forced_cost_norm: Optional[observation_t] = forced_cost_norm
        self.cost_norm: Optional[observation_t] = forced_cost_norm

    def reset(self) -> None:
        super().reset()
        self.cost_norm = self.forced_cost_norm

    def update(
        self,
        action: int,
        observations: List[observation_t],
        observation_view: ObservationView,
    ) -> float:
        if self.cost_norm is None:
            self.cost_norm = self.get_cost_norm(observation_view)
        return super().update(action, observations, observation_view) / self.cost_norm

    def get_cost_norm(self, observation_view: ObservationView) -> float:
        return observation_view[self.init_cost_function]


class BaselineImprovementNormalizedReward(NormalizedReward):
    """A cost function reward that is normalized to improvement made by a
    baseline approach.
    """

    __slots__ = ["baseline_cost_function"]

    def __init__(self, baseline_cost_function: str, **kwargs):
        super().__init__(**kwargs)
        self.baseline_cost_function: str = baseline_cost_function

    def get_cost_norm(self, observation_view: ObservationView) -> float:
        init_cost = observation_view[self.init_cost_function]
        baseline_cost = observation_view[self.baseline_cost_function]
        return min(init_cost - baseline_cost, 1)
