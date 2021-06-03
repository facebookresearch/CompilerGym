# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
from compiler_gym.datasets import Datasets
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import (
    ActionWrapper,
    CompilerEnvWrapper,
    ObservationWrapper,
    RewardWrapper,
)
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_wrapped_close(env: LlvmEnv):
    env = CompilerEnvWrapper(env)
    env.close()
    assert env.service is None


def test_wrapped_properties(env: LlvmEnv):
    """Test accessing the non-standard properties."""
    assert env.actions == []
    assert env.benchmark
    assert isinstance(env.datasets, Datasets)


def test_wrapped_fork_type(env: LlvmEnv):
    """Test forking a wrapper."""

    env = CompilerEnvWrapper(env)
    fkd = env.fork()
    try:
        assert isinstance(fkd, CompilerEnvWrapper)
    finally:
        fkd.close()


def test_wrapped_fork_subtype(env: LlvmEnv):
    """Test forking a wrapper subtype."""

    class MyWrapper(CompilerEnvWrapper):
        def __init__(self, env):
            super().__init__(env)

    env = MyWrapper(env)
    fkd = env.fork()
    try:
        assert isinstance(fkd, MyWrapper)
    finally:
        fkd.close()


def test_wrapped_fork_subtype_custom_constructor(env: LlvmEnv):
    """Test forking a wrapper with a custom constructor. This requires a custom
    fork() implementation."""

    class MyWrapper(CompilerEnvWrapper):
        def __init__(self, env, foo):
            super().__init__(env)
            self.foo = foo

        def fork(self):
            return MyWrapper(self.env.fork(), foo=self.foo)

    env = MyWrapper(env, foo=1)
    fkd = env.fork()
    try:
        assert isinstance(fkd, MyWrapper)
        assert fkd.foo == 1
    finally:
        fkd.close()


def test_wrapped_step_multi_step(env: LlvmEnv):
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    env.step([0, 0, 0])

    assert env.benchmark == "benchmark://cbench-v1/dijkstra"
    assert env.actions == [0, 0, 0]


def test_wrapped_action(env: LlvmEnv):
    class MyWrapper(ActionWrapper):
        def action(self, action):
            return action - 1

        def reverse_action(self, action):
            return action + 1

    env = MyWrapper(env)
    env.reset()
    env.step(1)
    env.step(2)

    assert env.actions == [0, 1]


def test_wrapped_observation(env: LlvmEnv):
    class MyWrapper(ObservationWrapper):
        def observation(self, observation):
            return isinstance(observation, str)
            return len(str)

    env.observation_space = "Ir"
    env = MyWrapper(env)
    assert env.reset() > 0
    observation, _, _, _ = env.step(0)
    assert observation > 0


def test_wrapped_reward(env: LlvmEnv):
    class MyWrapper(RewardWrapper):
        def reward(self, reward):
            return -5

    env.reward_space = "IrInstructionCount"
    env = MyWrapper(env)

    env.reset()
    _, reward, _, _ = env.step(0)
    assert reward == -5
    assert env.episode_reward == -5

    _, reward, _, _ = env.step(0)
    assert reward == -5
    assert env.episode_reward == -10


if __name__ == "__main__":
    main()
