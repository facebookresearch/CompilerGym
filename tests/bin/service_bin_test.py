# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:service."""
import sys

import gym
import pytest
from absl import flags

import compiler_gym
from compiler_gym.bin.service import print_service_capabilities
from compiler_gym.errors import EnvironmentNotSupported
from tests.test_main import main


@pytest.mark.parametrize("env_name", compiler_gym.COMPILER_GYM_ENVS)
@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="github.com/facebookresearch/CompilerGym/issues/459",
)
def test_print_service_capabilities_smoke_test(env_name: str):
    flags.FLAGS(["argv0"])
    try:
        with gym.make(env_name) as env:
            print_service_capabilities(env)
    except EnvironmentNotSupported:
        pass  # Environment not supported on this test platform.


if __name__ == "__main__":
    main()
