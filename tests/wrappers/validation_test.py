# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym.wrappers.llvm."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import ValidateBenchmarkAfterEveryStep
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_ValidateBenchmarkAfterEveryStep_valid(env: LlvmEnv):
    env.reset()

    type(env.benchmark).ivalidate = lambda *_: iter(())

    env = ValidateBenchmarkAfterEveryStep(env, reward_penalty=-5)
    _, reward, done, info = env.step(0)
    assert reward != -5
    assert not done
    assert "error_details" not in info


@pytest.mark.parametrize("reward_penalty", [-5, 10])
def test_ValidateBenchmarkAfterEveryStep_invalid(env: LlvmEnv, reward_penalty):
    env.reset()

    type(env.benchmark).ivalidate = lambda *_: iter(["Oh no!"])

    env = ValidateBenchmarkAfterEveryStep(env, reward_penalty=reward_penalty)
    _, reward, done, info = env.step(0)
    assert reward == reward_penalty
    assert done
    assert info["error_details"] == "Oh no!"


if __name__ == "__main__":
    main()
