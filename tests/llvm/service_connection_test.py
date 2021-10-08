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
from compiler_gym.service import ServiceError
from compiler_gym.service.connection import CompilerGymServiceConnection
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_DIM
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.fixture(scope="function", params=["local", "service"])
def env(request) -> CompilerEnv:
    # Redefine fixture to test both gym.make(...) and unmanaged service
    # connections.
    if request.param == "local":
        with gym.make("llvm-v0") as env:
            yield env
    else:
        service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
        try:
            with LlvmEnv(service=service.connection.url) as env:
                yield env
        finally:
            service.close()


def test_service_env_dies_reset(env: CompilerEnv):
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    env.reset("cbench-v1/crc32")

    # Kill the service. Note killing the service for a ManagedConnection will
    # result in a ServiceError because we have not ended the session we started
    # with env.reset() above. For UnmanagedConnection, this error will not be
    # raised.
    try:
        env.service.close()
    except ServiceError as e:
        assert "Service exited with returncode " in str(e)

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
