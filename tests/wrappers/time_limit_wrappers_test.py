# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import TimeLimit

# from gym.wrappers import TimeLimit
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_wrapped_close(env: LlvmEnv):
    env = TimeLimit(env, max_episode_steps=5)
    env.close()
    assert env.service is None


def test_wrapped_fork_type(env: LlvmEnv):
    env = TimeLimit(env, max_episode_steps=5)
    fkd = env.fork()
    try:
        assert isinstance(fkd, TimeLimit)
    finally:
        fkd.close()


def test_wrapped_step_multi_step(env: LlvmEnv):
    env = TimeLimit(env, max_episode_steps=5)
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    env.step([0, 0, 0])

    assert env.benchmark == "benchmark://cbench-v1/dijkstra"
    assert env.actions == [0, 0, 0]


def test_time_limit_reached(env: LlvmEnv):
    env = TimeLimit(env, max_episode_steps=3)

    env.reset()
    _, _, done, info = env.step(0)
    assert not done, info
    _, _, done, info = env.step(0)
    assert not done, info
    _, _, done, info = env.step(0)
    assert done, info
    assert info["TimeLimit.truncated"], info

    _, _, done, info = env.step(0)
    assert done, info
    assert info["TimeLimit.truncated"], info


def test_time_limit_fork(env: LlvmEnv):
    """Check that the time limit state is copied on fork()."""
    env = TimeLimit(env, max_episode_steps=3)

    env.reset()
    _, _, done, info = env.step(0)  # 1st step
    assert not done, info

    fkd = env.fork()
    try:
        _, _, done, info = env.step(0)  # 2nd step
        assert not done, info
        _, _, done, info = fkd.step(0)  # 2nd step
        assert not done, info

        _, _, done, info = env.step(0)  # 3rd step
        assert done, info
        _, _, done, info = fkd.step(0)  # 3rd step
        assert done, info
    finally:
        fkd.close()


if __name__ == "__main__":
    main()
