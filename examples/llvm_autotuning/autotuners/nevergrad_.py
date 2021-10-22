# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import lru_cache
from time import time
from typing import Tuple

import nevergrad as ng
from llvm_autotuning.optimization_target import OptimizationTarget

from compiler_gym.envs import CompilerEnv


def nevergrad(
    env: CompilerEnv,
    optimization_target: OptimizationTarget,
    search_time_seconds: int,
    seed: int,
    episode_length: int = 100,
    optimizer: str = "DiscreteLenglerOnePlusOne",
    **kwargs
) -> None:
    """Optimize an environment using nevergrad.

    Nevergrad is a gradient-free optimization platform that provides
    implementations of various black box optimizations techniques:

        https://facebookresearch.github.io/nevergrad/
    """
    if optimization_target == OptimizationTarget.RUNTIME:

        def calculate_negative_reward(actions: Tuple[int]) -> float:
            env.reset()
            env.step(actions)
            return -env.episode_reward

    else:
        # Only cache the deterministic non-runtime rewards.
        @lru_cache(maxsize=int(1e4))
        def calculate_negative_reward(actions: Tuple[int]) -> float:
            env.reset()
            env.step(actions)
            return -env.episode_reward

    params = ng.p.Choice(
        choices=range(env.action_space.n),
        repetitions=episode_length,
        deterministic=True,
    )
    params.random_state.seed(seed)

    optimizer_class = getattr(ng.optimizers, optimizer)
    optimizer = optimizer_class(parametrization=params, budget=1, num_workers=1)

    end_time = time() + search_time_seconds
    while time() < end_time:
        x = optimizer.ask()
        optimizer.tell(x, calculate_negative_reward(x.value))

    # Get best solution and replay it.
    recommendation = optimizer.provide_recommendation()
    env.reset()
    env.step(recommendation.value)
