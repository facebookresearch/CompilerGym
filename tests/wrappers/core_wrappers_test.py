# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
import pytest
from pytest import warns

from compiler_gym.datasets import Datasets
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import ActionWrapper, CompilerEnvWrapper
from compiler_gym.wrappers import ObservationWrapper as CoreObservationWrapper
from compiler_gym.wrappers import RewardWrapper as CoreRewardWrapper
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


class ObservationWrapper(CoreObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def convert_observation(self, observation):
        return observation


class RewardWrapper(CoreRewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def convert_reward(self, reward):
        return reward


@pytest.fixture(
    scope="module",
    params=[
        ActionWrapper,
        CompilerEnvWrapper,
        ObservationWrapper,
        RewardWrapper,
    ],
)
def wrapper_type(request):
    """A test fixture that yields one of the CompilerGym wrapper types."""
    return request.param


def test_wrapped_close(env: LlvmEnv, wrapper_type):
    env = wrapper_type(env)
    env.close()
    assert env.service is None


def test_wrapped_properties(env: LlvmEnv, wrapper_type):
    """Test accessing the non-standard properties."""
    with wrapper_type(env) as env:
        assert env.actions == []
        assert env.benchmark
        assert isinstance(env.datasets, Datasets)


def test_wrapped_fork_type(env: LlvmEnv, wrapper_type):
    """Test forking a wrapper."""

    env = wrapper_type(env)
    fkd = env.fork()
    try:
        assert isinstance(fkd, wrapper_type)
    finally:
        fkd.close()


def test_wrapped_fork_subtype(env: LlvmEnv, wrapper_type):
    """Test forking a wrapper subtype."""

    class MyWrapper(wrapper_type):
        def __init__(self, env):
            super().__init__(env)

    env = MyWrapper(env)
    fkd = env.fork()
    try:
        assert isinstance(fkd, MyWrapper)
    finally:
        fkd.close()


def test_wrapped_fork_subtype_custom_constructor(env: LlvmEnv, wrapper_type):
    """Test forking a wrapper with a custom constructor. This requires a custom
    fork() implementation."""

    class MyWrapper(wrapper_type):
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
    """Test passing a list of actions to step()."""
    env = CompilerEnvWrapper(env)
    env.reset()
    env.multistep([0, 0, 0])

    assert env.actions == [0, 0, 0]


def test_wrapped_step_custom_args(env: LlvmEnv, wrapper_type):
    """Test passing the custom CompilerGym step() keyword arguments."""

    class MyWrapper(wrapper_type):
        def convert_observation(self, observation):
            return observation  # pass thru

        def action(self, action):
            return action  # pass thru

        def convert_reward(self, reward):
            return reward

    env = MyWrapper(env)
    env.reset()
    (ir, ic), (icr, icroz), _, _ = env.multistep(
        actions=[0, 0, 0],
        observation_spaces=["Ir", "IrInstructionCount"],
        reward_spaces=["IrInstructionCount", "IrInstructionCountOz"],
    )
    assert isinstance(ir, str)
    assert isinstance(ic, int)
    assert isinstance(icr, float)
    assert isinstance(icroz, float)

    assert env.unwrapped.observation.spaces["Ir"].space.contains(ir)
    assert env.unwrapped.observation.spaces["IrInstructionCount"].space.contains(ic)


def test_wrapped_benchmark(env: LlvmEnv, wrapper_type):
    """Test that benchmark property has expected values."""

    class MyWrapper(wrapper_type):
        def convert_observation(self, observation):
            return observation  # pass thru

    env.observation_space = "Ir"
    env = MyWrapper(env)

    ir_a = env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"

    ir_b = env.reset(benchmark="benchmark://cbench-v1/qsort")
    assert env.benchmark == "benchmark://cbench-v1/qsort"

    # Check that the observations for different benchmarks are different.
    assert ir_a != ir_b


def test_wrapped_set_benchmark(env: LlvmEnv, wrapper_type):
    """Test that the benchmark attribute can be set on wrapped classes."""

    class MyWrapper(wrapper_type):
        def convert_observation(self, observation):
            return observation  # pass thru

    env = MyWrapper(env)

    # Set the benchmark attribute and check that it propagates.
    env.benchmark = "benchmark://cbench-v1/dijkstra"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/dijkstra"

    # Repeat again for a different benchmark.
    with warns(
        UserWarning,
        match=r"Changing the benchmark has no effect until reset\(\) is called",
    ):
        env.benchmark = "benchmark://cbench-v1/crc32"
    env.reset()
    assert env.benchmark == "benchmark://cbench-v1/crc32"


def test_wrapped_env_in_episode(env: LlvmEnv, wrapper_type):
    class MyWrapper(wrapper_type):
        def convert_observation(self, observation):
            return observation

    env = MyWrapper(env)
    assert not env.in_episode

    env.reset()
    assert env.in_episode


def test_wrapped_env_changes_default_spaces(env: LlvmEnv, wrapper_type):
    """Test when an environment wrapper changes the default observation and reward spaces."""

    class MyWrapper(wrapper_type):
        def __init__(self, env: LlvmEnv):
            super().__init__(env)
            self.env.observation_space = "Autophase"
            self.env.reward_space = "IrInstructionCount"

        def convert_observation(self, observation):
            return observation  # pass thru

    env = MyWrapper(env)
    assert env.observation_space.shape == (56,)
    assert env.observation_space_spec.id == "Autophase"
    assert env.reward_space.name == "IrInstructionCount"

    observation = env.reset()
    assert env.observation_space.contains(observation)


def test_wrapped_env_change_spaces(env: LlvmEnv, wrapper_type):
    """Test changing the observation and reward spaces on a wrapped environment."""

    class MyWrapper(wrapper_type):
        def convert_observation(self, observation):
            return observation  # pass thru

        def convert_reward(self, reward):
            return reward  # pass thru

    env = MyWrapper(env)

    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"

    assert env.observation_space.shape == (56,)
    assert env.observation_space_spec.id == "Autophase"
    assert env.reward_space.name == "IrInstructionCount"


def test_wrapped_action(mocker, env: LlvmEnv):
    class MyWrapper(ActionWrapper):
        def action(self, action):
            return action - 1

        def reverse_action(self, action):
            return action + 1

    env = MyWrapper(env)
    mocker.spy(env, "action")

    env.reset()
    env.step(1)
    env.step(2)

    assert env.action.call_count == 2  # pylint: disable=no-member
    assert env.actions == [0, 1]


def test_wrapped_observation(mocker, env: LlvmEnv):
    """Test using an ObservationWrapper that returns the length of the Ir string."""

    class MyWrapper(ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = "Ir"

        def convert_observation(self, observation):
            return len(observation)

    env = MyWrapper(env)

    assert env.reset() > 0
    observation, _, _, _ = env.step(0)

    assert observation > 0


def test_wrapped_observation_missing_definition(env: LlvmEnv):
    with pytest.raises(TypeError):
        env = CoreObservationWrapper(env)


def test_wrapped_reward(env: LlvmEnv):
    class MyWrapper(RewardWrapper):
        def convert_reward(self, reward):
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
