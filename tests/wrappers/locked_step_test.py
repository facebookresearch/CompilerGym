# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.wrappers.locked_step."""
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import LockedStep
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


class MockLock:
    def __init__(self):
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self):
        self.acquire_count += 1

    def release(self):
        self.release_count += 1

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args):
        self.release()


def test_wrapped_acquire_count(env: LlvmEnv):
    lock = MockLock()

    env = LockedStep(env, lock=lock)

    env.reset()
    assert lock.acquire_count == 1
    assert lock.release_count == 1

    env.step(env.action_space.sample())
    assert lock.acquire_count == 2
    assert lock.release_count == 2

    env.observation["IrInstructionCount"]
    assert lock.acquire_count == 2
    assert lock.release_count == 2

    env.reward["IrInstructionCount"]
    assert lock.acquire_count == 2
    assert lock.release_count == 2

    fkd = env.fork()
    try:
        assert lock.acquire_count == 3
        assert lock.release_count == 3
        assert fkd.lock.acquire_count == 3
        assert fkd.lock.release_count == 3
    finally:
        fkd.close()


def test_wrapped_default_lock(env: LlvmEnv):
    env = LockedStep(env)
    env.reset()
    env.step(env.action_space.sample())
    env.observation["IrInstructionCount"]
    env.reward["IrInstructionCount"]
    fkd = env.fork()
    fkd.step(env.action_space.sample())
    fkd.close()


if __name__ == "__main__":
    main()
