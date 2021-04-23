# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import gym
import numpy as np
import pytest

import compiler_gym  # noqa Register environments.
from compiler_gym.envs import CompilerEnv, llvm
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.service.connection import CompilerGymServiceConnection
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_DIM
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.fixture(scope="function", params=["local", "service"])
def env(request) -> CompilerEnv:
    # Redefine fixture to test both gym.make(...) and unmanaged service
    # connections.
    if request.param == "local":
        env = gym.make("llvm-v0")
        try:
            yield env
        finally:
            env.close()
    else:
        service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
        env = LlvmEnv(service=service.connection.url)
        try:
            yield env
        finally:
            env.close()
            service.close()


def test_service_env_dies_reset(env: CompilerEnv):
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env.reset("cbench-v1/crc32")

    # Kill the service.
    env.service.close()

    # Check that the environment doesn't fall over.
    observation, reward, done, info = env.step(0)
    assert done, info["error_details"]
    assert not env.in_episode

    # Check that default values are returned.
    np.testing.assert_array_equal(observation, np.zeros(AUTOPHASE_FEATURE_DIM))
    assert reward == 0

    # Reset the environment and check that it works.
    env.reset(benchmark="cbench-v1/crc32")
    assert env.in_episode

    observation, reward, done, info = env.step(0)
    assert not done, info["error_details"]
    assert observation is not None
    assert reward is not None


if __name__ == "__main__":
    main()
