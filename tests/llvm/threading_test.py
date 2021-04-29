# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
from threading import Thread
from typing import List

import gym

from compiler_gym import CompilerEnv
from tests.test_main import main


class ThreadedWorker(Thread):
    """Create an environment and run through a set of actions in a background thread."""

    def __init__(self, env_name: str, benchmark: str, actions: List[int]):
        super().__init__()
        self.done = False
        self.env_name = env_name
        self.benchmark = benchmark
        self.actions = actions
        assert actions

    def run(self) -> None:
        env = gym.make(self.env_name, benchmark=self.benchmark)
        env.reset()

        for action in self.actions:
            self.observation, self.reward, done, self.info = env.step(action)
            assert not done, self.info["error_details"]

        self.done = True
        env.close()


class ThreadedWorkerWithEnv(Thread):
    """Create an environment and run through a set of actions in a background thread."""

    def __init__(self, env: CompilerEnv, actions: List[int]):
        super().__init__()
        self.done = False
        self.env = env
        self.actions = actions
        assert actions

    def run(self) -> None:
        for action in self.actions:
            self.observation, self.reward, done, self.info = self.env.step(action)
            assert not done, self.info["error_details"]

        self.done = True


def test_running_environment_in_background_thread():
    """Test launching and running an LLVM environment in a background thread."""
    thread = ThreadedWorker(
        env_name="llvm-autophase-ic-v0",
        benchmark="cbench-v1/crc32",
        actions=[0, 0, 0],
    )
    thread.start()
    thread.join(timeout=10)

    assert thread.done
    assert thread.observation is not None
    assert isinstance(thread.reward, float)
    assert thread.info


def test_moving_environment_to_background_thread():
    """Test running an LLVM environment from a background thread. The environment
    is made in the main thread and used in the background thread.
    """
    env = gym.make("llvm-autophase-ic-v0")
    env.reset(benchmark="cbench-v1/crc32")

    thread = ThreadedWorkerWithEnv(env=env, actions=[0, 0, 0])
    thread.start()
    thread.join(timeout=10)

    assert thread.done
    assert thread.observation is not None
    assert isinstance(thread.reward, float)
    assert thread.info

    assert env.in_episode
    env.close()


if __name__ == "__main__":
    main()
