# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Validate environment states."""
import math
import multiprocessing
import multiprocessing.pool
from typing import Callable, Iterable, List, NamedTuple, Optional, cast

import gym

from compiler_gym.envs.compiler_env import CompilerEnv, CompilerEnvState
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets import LLVM_BENCHMARK_VALIDATION_CALLBACKS
from compiler_gym.spaces import Commandline
from compiler_gym.util.timer import Timer


class ValidationResult(NamedTuple):
    """The result of validating a compiler state."""

    state: CompilerEnvState
    """The compiler environment state that was validated."""

    reward_validated: bool
    """Whether the reward that was recorded in the original state was validated."""

    actions_replay_failed: bool
    """Whether the commandline was unable to be reproduced."""

    reward_validation_failed: bool
    """Whether the validated reward differed from the original state."""

    benchmark_semantics_validated: bool
    """Whether the semantics of the benchmark were validated."""

    benchmark_semantics_validation_failed: bool
    """Whether the semantics of the benchmark were found to have changed."""

    walltime: float
    """The wall time in seconds that the validation took."""

    error_details: str = ""
    """A description of any validation errors."""

    @property
    def success(self) -> bool:
        """Whether validation succeeded."""
        return not self.failed

    @property
    def failed(self) -> bool:
        """Whether validation failed."""
        return (
            self.actions_replay_failed
            or self.reward_validation_failed
            or self.benchmark_semantics_validation_failed
        )

    def __repr__(self):
        if self.failed:
            return f"❌  {self.state.benchmark}  {self.error_details}"
        elif self.state.reward is None:
            return f"✅  {self.state.benchmark}"
        else:
            return f"✅  {self.state.benchmark}  {self.state.reward:.4f}"


def _llvm_replay_commandline(env: LlvmEnv, commandline: str) -> Optional[float]:
    """Replay the sequence of actions given by a commandline."""

    # Strip the decorative elements that LlvmEnv.commandline() adds.
    if not commandline.startswith("opt ") or not commandline.endswith(
        " input.bc -o output.bc"
    ):
        raise ValueError(f"Invalid commandline: `{commandline}`")
    commandline = commandline[len("opt ") : -len(" input.bc -o output.bc")]

    actions = cast(Commandline, env.action_space).from_commandline(commandline)
    for action in actions:
        _, _, done, info = env.step(action)
        if done:
            raise OSError(
                f"Environment terminated with error: `{info.get('error_details')}`"
            )
    return env.episode_reward


def validate_state(env: CompilerEnv, state: CompilerEnvState) -> ValidationResult:
    """Validate a :class:`CompilerEnvState <compiler_gym.envs.CompilerEnvState>`.

    :param env: A compiler environment.
    :param state: The environment state to validate.
    :return: A :class:`ValidationResult <compiler_gym.ValidationResult>` instance.
    """
    error_messages = []
    validation = {
        "state": state,
        "actions_replay_failed": False,
        "reward_validated": False,
        "reward_validation_failed": False,
        "benchmark_semantics_validated": False,
        "benchmark_semantics_validation_failed": False,
    }

    if state.reward is not None and env.reward_space is None:
        raise ValueError("Reward space not specified")

    with Timer() as walltime:
        env.reset(benchmark=state.benchmark)
        # Use a while loop here so that we can `break` early out of the
        # validation process in case a step fails.
        while True:
            try:
                reward = _llvm_replay_commandline(env, state.commandline)
            except (ValueError, OSError) as e:
                validation["actions_replay_failed"] = True
                error_messages.append(str(e))
                break

            if state.reward is not None and env.reward_space.deterministic:
                validation["reward_validated"] = True
                # If reward deviates from the expected amount record the
                # error but continue with the remainder of the validation.
                if not math.isclose(reward, state.reward, rel_tol=1e-5, abs_tol=1e-10):
                    validation["reward_validation_failed"] = True
                    error_messages.append(
                        f"Expected reward {state.reward:.4f} but received reward {reward:.4f}"
                    )

            validate_semantics = LLVM_BENCHMARK_VALIDATION_CALLBACKS.get(
                state.benchmark
            )
            if validate_semantics:
                validation["benchmark_semantics_validated"] = True
                semantics_error = validate_semantics(env)
                if semantics_error:
                    validation["benchmark_semantics_validation_failed"] = True
                    error_messages.append(semantics_error)

            # Finished all checks, break the loop.
            break

    return ValidationResult(
        walltime=walltime.time, error_details="\n".join(error_messages), **validation
    )


def _validate_states_worker(args) -> ValidationResult:
    reward_space, state = args
    env = gym.make("llvm-v0", reward_space=reward_space)
    try:
        result = validate_state(env, state)
    finally:
        env.close()
    return result


def validate_states(
    make_env: Callable[[], CompilerEnv],
    states: Iterable[CompilerEnvState],
    datasets: Optional[List[str]] = None,
    nproc: Optional[int] = None,
) -> Iterable[ValidationResult]:
    """A parallelized implementation of
    :func:`validate_state() <compiler_gym.validate_state>` for batched
    validation.

    :param make_env: A callback which instantiates a compiler environment.
    :param states: A sequence of compiler environment states to validate.
    :param datasets: An optional list of datasets that are required.
    :param nproc: The number of parallel worker processes to run.
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
        yield from pool.imap_unordered(
            _validate_states_worker, [(reward_space_name, r) for r in states]
        )
