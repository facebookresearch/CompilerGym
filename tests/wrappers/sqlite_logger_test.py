# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/wrappers."""
import pytest

from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.wrappers import CompilerEnvWrapper, SynchronousSqliteLogger
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_SynchronousSqliteLogger_creates_file(env: LlvmEnv, tmp_path):
    db_path = tmp_path / "example.db"
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env = SynchronousSqliteLogger(env, db_path)
    env.reset()
    env.step(0)
    env.flush()
    assert db_path.is_file()


def test_SynchronousSqliteLogger_requires_llvm_env(tmp_path):
    with pytest.raises(TypeError, match="Requires LlvmEnv base environment"):
        SynchronousSqliteLogger(1, tmp_path / "example.db")


def test_SynchronousSqliteLogger_wrapped_env(env: LlvmEnv, tmp_path):
    env = CompilerEnvWrapper(env)
    env = SynchronousSqliteLogger(env, tmp_path / "example.db")
    env.reset()


if __name__ == "__main__":
    main()
