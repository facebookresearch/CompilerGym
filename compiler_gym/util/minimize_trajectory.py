# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines utilities for minimizing trajectories.

A trajectory is the sequence of actions applied to an environment. The goal of
trajectory minimization is to identify the shortest subregion of a trajectory
such that some hypothesis still holds. A a hypothesis is a boolean test on an
environment, for example, a hypothesis could be that :code:`env.validate()`
returns an error.
"""
import logging
import random
from math import ceil, log
from typing import Callable, Iterable

from compiler_gym.util.truncate import truncate


class MinimizationError(OSError):
    """Error raised if trajectory minimization fails."""


# A hypothesis is a callback that accepts as input an enivornment in a given
# state returns true if a particular hypothesis holds, else false.
Hypothesis = Callable[["CompilerEnv"], bool]  # noqa: F821


def environment_validation_fails(env: "CompilerEnv") -> bool:  # noqa: F821
    """A hypothesis that holds true if environment validation fails."""
    validation_result = env.validate()
    logging.debug(truncate(str(validation_result), max_lines=1, max_line_len=120))
    return not validation_result.okay()


def _apply_and_test(env, actions, hypothesis, flakiness) -> bool:
    """Run specific actions on environment and return whether hypothesis holds."""
    env.reset(benchmark=env.benchmark)
    for _ in range(flakiness):
        logging.debug("Applying %d actions ...", len(actions))
        _, _, done, info = env.step(actions)
        if done:
            raise MinimizationError(
                f"Failed to replay actions: {info.get('error_details', '')}"
            )
        logging.debug("Applied %d actions", len(actions))
        if hypothesis(env):
            return True
    return False


def bisect_trajectory(
    env: "CompilerEnv",  # noqa: F821
    hypothesis: Hypothesis = environment_validation_fails,
    reverse: bool = False,
    flakiness: int = 1,
) -> Iterable["CompilerEnv"]:  # noqa: F821
    """Run a binary search to remove the suffix or prefix of a trjectory.

    Requires worst-case O(log n) evaluation rounds, where n is the length of the
    trajectory.

    :param env: An environment whose action trajectory should be minimized.
    :param hypothesis: The hypothesis that is used to determine if a trajectory
        is valid. A callback that accepts as argument the :code:`env:`
        instance and returns true if the hypothesis holds, else false. The
        hypothesis must hold on the initial trajectory.
    :param reverse: If :code:`True`, minimize the trajectory from the front
        (i.e. the prefix). Else, minimization occurs form the back (i.e. the
        suffix).
    :param flakiness: The maximum number of times the hypothesis is repeated
        to check if it holds. If the hypothesis returns :code:`True` within this
        many iterations, it is said to hold. It needs to only return
        :code:`True` once.
    :returns: A generator that yields the input environment every time the
        trajectory is successfully reduced.
    :raises MinimizationError: If the environment action replay fails, or if
        the hypothesis does not hold on the initial trajectory.
    """

    def apply_and_test(actions):
        return _apply_and_test(env, actions, hypothesis, flakiness)

    all_actions = env.actions.copy()
    # No actions to minimize.
    if not all_actions:
        return env

    logging.info(
        "%sisecting sequence of %d actions",
        "Reverse b" if reverse else "B",
        len(all_actions),
    )
    if not apply_and_test(all_actions):
        raise MinimizationError(
            "Hypothesis failed on the initial state! The hypothesis must hold for the first state."
        )

    left = 0
    right = len(all_actions) - 1
    step = 0
    while right >= left:
        step += 1
        remaining_steps = int(log(max(right - left, 1), 2))
        mid = left + ((right - left) // 2)
        logging.debug(
            "Bisect step=%d, left=%d, right=%d, mid=%d", step, left, right, mid
        )

        actions = all_actions[mid:] if reverse else all_actions[:mid]
        if apply_and_test(actions):
            logging.info(
                "🟢 Hypothesis holds at num_actions=%d, remaining bisect steps=%d",
                mid,
                remaining_steps,
            )
            yield env
            if reverse:
                left = mid + 1
            else:
                right = mid - 1
        else:
            logging.info(
                "🔴 Hypothesis does not hold at num_actions=%d, remaining bisect steps=%d",
                mid,
                remaining_steps,
            )
            if reverse:
                right = mid - 1
            else:
                left = mid + 1

    mid = max(left, right) - 1 if reverse else min(left, right) + 1
    if (reverse and mid < 0) or (not reverse and mid >= len(all_actions)):
        actions = all_actions
        logging.info("Failed to reduce trajectory length using bisection")
    else:
        actions = all_actions[mid:] if reverse else all_actions[:mid]
        logging.info(
            "Determined that action %d of %d is the first at which the hypothesis holds: %s",
            mid,
            len(all_actions),
            env.action_space.flags[all_actions[mid]],
        )

    if not apply_and_test(actions):
        raise MinimizationError("Post-bisect sanity check failed!")

    yield env


def random_minimization(
    env: "CompilerEnv",  # noqa: F821
    hypothesis: Hypothesis = environment_validation_fails,
    num_steps_ratio_multiplier: float = 5,
    init_discard_ratio: float = 0.75,
    discard_ratio_decay: float = 0.75,
    min_trajectory_len: int = 5,
    flakiness: int = 1,
) -> Iterable["CompilerEnv"]:  # noqa: F821
    """Run an iterative process of randomly removing actions to minimize a
    trajectory.

    For each round of minimization, a number of actions are discarded randomly
    and the hypothesis is tested. If the hypothesis still holds with those
    actions removed, the minimization proceeds. Else the actions are re-inserted
    into the trajectory and a new set of actions are removed. After a failure

    Performs up to O(num_steps_ratio_multiplier * log n) evaluation rounds,
    where n is the length of the trajectory.

    :param env: An environment whose action trajectory should be minimized.
    :param hypothesis: The hypothesis that is used to determine if a trajectory
        is valid. A callback that accepts as argument the :code:`env:`
        instance and returns true if the hypothesis holds, else false. The
        hypothesis must hold on the initial trajectory.
    :param num_steps_ratio_multiplier: A multiplier for the number of rounds of
        minimization to perform, using log(n) the length of the trajectory as
        the factor.
    :param init_discard_ratio: The number of actions that will be randomly
        discarded, as a multiplier of the length of the trajectory.
    :param discard_ratio_decay: The ratio of decay for the discard ratio on
        failure.
    :param min_trajectory_len: The minimum number of actions in the trajectory
        for minimization to run. If the trajectory contains fewer than this many
        actions, minimization stops.
    :param flakiness: The maximum number of times the hypothesis is repeated
        to check if it holds. If the hypothesis returns :code:`True` within this
        many iterations, it is said to hold. It needs to only return
        :code:`True` once.
    :returns: A generator that yields the input environment every time the
        trajectory is successfully reduced.
    :raises MinimizationError: If the environment action replay fails, or if
        the hypothesis does not hold on the initial trajectory.
    """

    def apply_and_test(actions):
        return _apply_and_test(env, actions, hypothesis, flakiness)

    actions = env.actions.copy()
    if not apply_and_test(actions):
        raise MinimizationError(
            "Hypothesis failed on the initial state! The hypothesis must hold for the first state."
        )

    max_num_steps = int(log(len(actions), 2) * num_steps_ratio_multiplier)

    num_steps = 0
    discard_ratio = init_discard_ratio
    while len(actions) >= min_trajectory_len and num_steps < max_num_steps:
        num_steps += 1
        num_to_remove = int(ceil(len(actions) * discard_ratio))
        candidate_actions = actions.copy()
        # Delete actions randomly.
        for _ in range(num_to_remove):
            del candidate_actions[random.randint(0, len(candidate_actions) - 1)]
        if apply_and_test(candidate_actions):
            logging.info(
                "🟢 Hypothesis holds with %s of %s actions randomly removed, continuing",
                num_to_remove,
                len(actions),
            )
            actions = candidate_actions
            discard_ratio = init_discard_ratio
            yield env
        else:
            logging.info(
                "🔴 Hypothesis does not hold with %s of %s actions randomly removed, rolling back",
                num_to_remove,
                len(actions),
            )
            discard_ratio *= discard_ratio_decay
            if num_to_remove == 1:
                logging.info(
                    "Terminating random minimization after failing with only a single action removed"
                )
                break

    if not apply_and_test(actions):
        raise MinimizationError("Post-minimization sanity check failed!")

    yield env


def minimize_trajectory_iteratively(
    env: "CompilerEnv",  # noqa: F821
    hypothesis: Hypothesis = environment_validation_fails,
    flakiness: int = 1,
) -> Iterable["CompilerEnv"]:  # noqa: F821
    """Minimize a trajectory by remove actions, one at a time, until a minimal
    trajectory is reached.

    Performs up to O(n * n / 2) evaluation rounds, where n is the length of the
    trajectory.

    :param env: An environment whose action trajectory should be minimized.
    :param hypothesis: The hypothesis that is used to determine if a trajectory
        is valid. A callback that accepts as argument the :code:`env:`
        instance and returns true if the hypothesis holds, else false. The
        hypothesis must hold on the initial trajectory.
    :param flakiness: The maximum number of times the hypothesis is repeated
        to check if it holds. If the hypothesis returns :code:`True` within this
        many iterations, it is said to hold. It needs to only return
        :code:`True` once.
    :returns: A generator that yields the input environment every time the
        trajectory is successfully reduced.
    :raises MinimizationError: If the environment action replay fails, or if
        the hypothesis does not hold on the initial trajectory.
    """

    def apply_and_test(actions):
        return _apply_and_test(env, actions, hypothesis, flakiness)

    all_actions = env.actions.copy()
    init_num_actions = len(all_actions)
    if not all_actions:  # Nothing to minimize.
        return

    if not apply_and_test(all_actions):
        raise MinimizationError(
            "Hypothesis failed on the initial state! The hypothesis must hold for the first state."
        )

    pass_num = 0
    actions_removed = 0
    action_has_been_pruned = True
    # Outer loop. Repeat iterative reduction until no change is made.
    while action_has_been_pruned and len(all_actions) > 1:
        pass_num += 1
        action_has_been_pruned = False
        action_mask = [True] * len(all_actions)
        logging.info("Minimization pass on sequence of %d actions", len(all_actions))

        # Inner loop. Go through every action and see if it can be removed.
        for i in range(len(action_mask)):
            action_mask[i] = False
            action_name = env.action_space.flags[all_actions[i]]
            actions = [action for action, mask in zip(all_actions, action_mask) if mask]
            if apply_and_test(actions):
                logging.info(
                    "🟢 Hypothesis holds with action %s removed, %d actions remaining",
                    action_name,
                    sum(action_mask),
                )
                action_has_been_pruned = True
                actions_removed += 1
                yield env
            else:
                action_mask[i] = True
                logging.info(
                    "🔴 Hypothesis does not hold with action %s removed, %d actions remaining",
                    action_name,
                    sum(action_mask),
                )

        all_actions = [action for action, mask in zip(all_actions, action_mask) if mask]

    logging.info(
        "Minimization halted after %d passes, %d of %d actions removed",
        pass_num,
        actions_removed,
        init_num_actions,
    )
    if not apply_and_test(all_actions):
        raise ValueError("Post-bisect sanity check failed!")

    yield env
