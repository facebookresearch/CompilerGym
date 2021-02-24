# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Validate environment states."""
import multiprocessing
import multiprocessing.pool
from typing import Callable, Iterable, List, Optional

import gym

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.validation_result import ValidationResult


def _validate_states_worker(args) -> ValidationResult:
    reward_space, state = args
    env = gym.make("llvm-v0", reward_space=reward_space)
    try:
        result = env.validate(state)
    finally:
        env.close()
    return result


def validate_states(
    make_env: Callable[[], CompilerEnv],
    states: Iterable[CompilerEnvState],
    datasets: Optional[List[str]] = None,
    nproc: Optional[int] = None,
    inorder: bool = False,
) -> Iterable[ValidationResult]:
    """A parallelized implementation of
    :meth:`env.validate() <compiler_gym.envs.CompilerEnv.validate>` for batched
    validation.

    :param make_env: A callback which instantiates a compiler environment.
    :param states: A sequence of compiler environment states to validate.
    :param datasets: An optional list of datasets that are required.
    :param nproc: The number of parallel worker processes to run.
    :param inorder: Whether to return results in the order they were provided,
        or in the order that they are available.
    :return: An iterator over validation results. The order of results may
        differ from the input states.
    """
    env = make_env()
    try:
        if not isinstance(env, LlvmEnv):
            raise ValueError("Only LLVM environment is supported for validation.")

        # Ensure that the required datasets are available.
        env.require_datasets(datasets)
        reward_space_name: str = env.reward_space.id if env.reward_space else None
    finally:
        env.close()

    with multiprocessing.Pool(processes=nproc) as pool:
        if nproc == 1:
            map_func = map
        elif inorder:
            map_func = pool.imap
        else:
            map_func = pool.imap_unordered

        yield from map_func(
            _validate_states_worker, [(reward_space_name, r) for r in states]
        )
