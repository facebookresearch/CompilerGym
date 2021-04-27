# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Validate environment states."""
import random
from concurrent.futures import as_completed
from typing import Callable, Iterable, Optional

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.util import thread_pool
from compiler_gym.validation_result import ValidationResult


def _validate_states_worker(
    make_env: Callable[[], CompilerEnv], state: CompilerEnvState
) -> ValidationResult:
    env = make_env()
    try:
        result = env.validate(state)
    finally:
        env.close()
    return result


def validate_states(
    make_env: Callable[[], CompilerEnv],
    states: Iterable[CompilerEnvState],
    nproc: Optional[int] = None,
    inorder: bool = False,
) -> Iterable[ValidationResult]:
    """A parallelized implementation of
    :meth:`env.validate() <compiler_gym.envs.CompilerEnv.validate>` for batched
    validation.

    :param make_env: A callback which instantiates a compiler environment.
    :param states: A sequence of compiler environment states to validate.
    :param nproc: The number of parallel worker processes to run.
    :param inorder: Whether to return results in the order they were provided,
        or in the order that they are available.
    :return: An iterator over validation results. The order of results may
        differ from the input states.
    """
    executor = thread_pool.get_thread_pool_executor()

    if nproc == 1:
        map_func = map
    elif inorder:
        map_func = executor.map
    else:
        # The validation function of benchmarks can vary wildly in computational
        # demands. Shuffle the order of states (unless explicitly asked for them
        # to be kept inorder) as crude load balancing for the case where
        # multiple states are provided for each benchmark.
        states = list(states)
        random.shuffle(states)

        def map_func(func, envs, states):
            futures = (
                executor.submit(func, env, state) for env, state in zip(envs, states)
            )
            return (r.result() for r in as_completed(futures))

    yield from map_func(_validate_states_worker, [make_env] * len(states), states)
