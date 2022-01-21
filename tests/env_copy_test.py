# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the copy() and deepcopy() operators on CompilerEnv."""
from copy import copy, deepcopy

import pytest

from compiler_gym.envs.llvm import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_forbidden_shallow_copy(env: LlvmEnv):
    """Test that shallow copy operator is explicitly forbidden."""
    with pytest.raises(
        TypeError,
        match=r"^CompilerEnv instances do not support shallow copies. Use deepcopy\(\)",
    ):
        copy(env)


def test_deep_copy(env: LlvmEnv):
    """Test that deep copy creates an independent copy."""
    env.reset()
    with deepcopy(env) as cpy:
        assert cpy.state == env.state
        env.step(env.action_space.sample())
        assert cpy.state != env.state


if __name__ == "__main__":
    main()
