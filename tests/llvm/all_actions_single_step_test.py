# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""

import numpy as np

from compiler_gym.envs import CompilerEnv
from compiler_gym.errors import ServiceError
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_DIM
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_step(env: CompilerEnv, action_name: str):
    """Run each action on a single benchmark."""
    env.reward_space = "IrInstructionCount"
    env.observation_space = "Autophase"
    env.reset(benchmark="cbench-v1/crc32")

    action = env.action_space.from_string(action_name)[0]

    observation, reward, done, _ = env.step(action)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (AUTOPHASE_FEATURE_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    try:
        env.close()
    except ServiceError as e:
        # env.close() will raise an error if the service terminated
        # ungracefully. In that case, the "done" flag should have been set.
        assert done, f"Service error was raised when 'done' flag not set: {e}"


if __name__ == "__main__":
    main()
