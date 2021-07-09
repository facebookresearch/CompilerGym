# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for LLVM session parameter handlers."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.service import SessionNotFound
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_send_param_before_reset(env: LlvmEnv):
    """Test that send_params() before reset() raises an error."""
    with pytest.raises(
        SessionNotFound, match=r"Must call reset\(\) before send_params\(\)"
    ):
        env.send_params(("test", "test"))


def test_send_param_unknown_key(env: LlvmEnv):
    """Test that send_params() raises an error when the key is not recognized."""
    env.reset()
    with pytest.raises(ValueError, match="Unknown parameter: unknown.key"):
        env.send_params(("unknown.key", ""))


if __name__ == "__main__":
    main()
