# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import Counter
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_Counter_reset(env: LlvmEnv):
    with Counter(env) as env:
        env.reset()
        assert env.counters == {
            "close": 0,
            "fork": 0,
            "reset": 1,
            "step": 0,
        }

        env.reset()
        assert env.counters == {
            "close": 0,
            "fork": 0,
            "reset": 2,
            "step": 0,
        }


def test_Counter_step(env: LlvmEnv):
    with Counter(env) as env:
        env.reset()
        env.step(0)
        assert env.counters == {
            "close": 0,
            "fork": 0,
            "reset": 1,
            "step": 1,
        }


def test_Counter_double_close(env: LlvmEnv):
    with Counter(env) as env:
        env.close()
        env.close()
        assert env.counters == {
            "close": 2,
            "fork": 0,
            "reset": 0,
            "step": 0,
        }

    # Implicit close in `with` statement.
    assert env.counters == {
        "close": 3,
        "fork": 0,
        "reset": 0,
        "step": 0,
    }


if __name__ == "__main__":
    main()
