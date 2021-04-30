# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test that LlvmEnv is compatible with OpenAI gym interface."""
import gym
import pytest

import compiler_gym  # noqa Register Environments
from compiler_gym.envs import CompilerEnv
from tests.test_main import main


@pytest.fixture(scope="function")
def env() -> CompilerEnv:
    env = gym.make("llvm-autophase-ic-v0")
    try:
        yield env
    finally:
        env.close()


def test_type_classes(env: CompilerEnv):
    assert isinstance(env, gym.Env)
    assert isinstance(env, CompilerEnv)
    assert isinstance(env.unwrapped, CompilerEnv)
    assert isinstance(env.action_space, gym.Space)
    assert isinstance(env.observation_space, gym.Space)
    assert isinstance(env.reward_range[0], float)
    assert isinstance(env.reward_range[1], float)


def test_optional_properties(env: CompilerEnv):
    assert "render.modes" in env.metadata
    assert env.spec


def test_contextmanager(env: CompilerEnv, mocker):
    mocker.spy(env, "close")
    assert env.close.call_count == 0
    with env:
        pass
    assert env.close.call_count == 1


def test_contextmanager_gym_make(mocker):
    with gym.make("llvm-v0") as env:
        mocker.spy(env, "close")
        assert env.close.call_count == 0
        with env:
            pass
        assert env.close.call_count == 1


def test_observation_wrapper(env: CompilerEnv):
    class WrappedEnv(gym.ObservationWrapper):
        def observation(self, observation):
            return "Hello"

    wrapped = WrappedEnv(env)
    observation = wrapped.reset()
    assert observation == "Hello"

    observation, _, _, _ = wrapped.step(0)
    assert observation == "Hello"


def test_reward_wrapper(env: CompilerEnv):
    class WrappedEnv(gym.RewardWrapper):
        def reward(self, reward):
            return 1

    wrapped = WrappedEnv(env)
    wrapped.reset()

    _, reward, _, _ = wrapped.step(0)
    assert reward == 1


if __name__ == "__main__":
    main()
