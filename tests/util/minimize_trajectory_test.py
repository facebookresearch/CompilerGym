# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:minimize_trajectory."""
import logging
import sys
from typing import List

import pytest

from compiler_gym.util import minimize_trajectory as mt
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# Verbose logging for tests.
logging.basicConfig(level=logging.DEBUG)


class MockActionSpace:
    """A mock action space for use by MockEnv."""

    def __init__(self, actions):
        self.flags = {a: str(a) for a in set(actions)}


class MockValidationResult:
    """A mock validation result for use by MockEnv."""

    def __init__(self, okay):
        self._okay = okay

    def okay(self):
        return self._okay


class MockEnv:
    """A mock environment for testing trajectory minimization."""

    def __init__(self, actions: List[int], validate=lambda env: True):
        self.original_trajectory = actions
        self.actions = actions.copy()
        self.validate = lambda: MockValidationResult(validate(self))
        self.benchmark = "benchmark"
        self.action_space = MockActionSpace(set(actions))

    def reset(self, benchmark):
        self.actions = []
        assert benchmark == self.benchmark

    def step(self, actions):
        for action in actions:
            assert action in self.original_trajectory
        self.actions += actions
        return None, None, False, {}


def make_hypothesis(val: int):
    """Create a hypothesis that checks if `val` is in actions."""

    def hypothesis(env):
        print("hypothesis?()", env.actions, val in env.actions, file=sys.stderr)
        return val in env.actions

    return hypothesis


@pytest.mark.parametrize("n", range(10))
def test_bisect_explicit_hypothesis(n: int):
    """Test that bisection chops off the tail."""
    env = MockEnv(actions=list(range(10)))
    list(mt.bisect_trajectory(env, make_hypothesis(n)))
    assert env.actions == list(range(n + 1))


@pytest.mark.parametrize("n", range(10))
def test_bisect_implicit_hypothesis(n: int):
    """Test bisection again but using the implicit hypothesis that
    env.validate() fails.
    """
    env = MockEnv(
        actions=list(range(10)), validate=lambda env: not make_hypothesis(n)(env)
    )
    list(mt.bisect_trajectory(env))
    assert env.actions == list(range(n + 1))


@pytest.mark.parametrize("n", range(10))
def test_reverse_bisect(n: int):
    """Test that reverse bisection chops off the prefix."""
    env = MockEnv(actions=list(range(10)))
    list(mt.bisect_trajectory(env, make_hypothesis(n), reverse=True))
    assert env.actions == list(range(n, 10))


def test_minimize_trajectory_iteratively():
    """Test that reverse bisection chops off the prefix."""
    env = MockEnv(actions=list(range(10)))

    minimized = [0, 3, 4, 5, 8, 9]

    def hypothesis(env):
        return all(x in env.actions for x in minimized)

    list(mt.minimize_trajectory_iteratively(env, hypothesis))
    assert env.actions == minimized


def test_minimize_trajectory_iteratively_no_effect():
    """Test that reverse bisection chops off the prefix."""
    env = MockEnv(actions=list(range(10)))

    minimized = list(range(10))

    def hypothesis(env):
        return env.actions == minimized

    list(mt.minimize_trajectory_iteratively(env, hypothesis))
    assert env.actions == minimized


def test_random_minimization():
    """Test that random minimization reduces trajectory."""
    env = MockEnv(actions=list(range(10)))

    minimized = [0, 1, 4]

    def hypothesis(env):
        return all(x in env.actions for x in minimized)

    list(mt.random_minimization(env, hypothesis))
    assert len(env.actions) <= 10
    assert len(env.actions) >= len(minimized)
    assert all(a in list(range(10)) for a in env.actions)


def test_random_minimization_no_effect():
    """Test random minimization when there's no improvement to be had."""
    env = MockEnv(actions=list(range(10)))

    minimized = list(range(10))

    def hypothesis(env):
        return env.actions == minimized

    list(mt.random_minimization(env, hypothesis))
    assert env.actions == minimized


def test_minimize_trajectory_iteratively_llvm_crc32(env):
    """Test trajectory minimization on a real environment."""
    env.reset(benchmark="cbench-v1/crc32")
    env.step(
        [
            env.action_space["-mem2reg"],
            env.action_space["-gvn"],
            env.action_space["-reg2mem"],
        ]
    )

    def hypothesis(env):
        return (
            env.action_space["-mem2reg"] in env.actions
            and env.action_space["-reg2mem"] in env.actions
        )

    list(mt.minimize_trajectory_iteratively(env, hypothesis))
    assert env.actions == [
        env.action_space["-mem2reg"],
        env.action_space["-reg2mem"],
    ]


if __name__ == "__main__":
    main()
