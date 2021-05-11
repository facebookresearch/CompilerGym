# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import compiler_gym
from compiler_gym.envs import LlvmEnv
from tests.test_main import main


def test_compiler_gym_make():
    """Test that compiler_gym.make() is equivalent to gym.make()."""
    with compiler_gym.make("llvm-v0") as env:
        assert isinstance(env, LlvmEnv)


if __name__ == "__main__":
    main()
