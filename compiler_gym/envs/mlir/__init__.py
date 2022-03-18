# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Register the MLIR environments."""

from compiler_gym.envs.mlir.mlir_env import MlirEnv
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

__all__ = [
    "MLIR_SERVICE_BINARY",
    "MlirEnv",
    "observation_spaces",
    "reward_spaces",
]

MLIR_SERVICE_BINARY = runfiles_path(
    "compiler_gym/envs/mlir/service/compiler_gym-mlir-service"
)


def _register_mlir_gym_service():
    """Register an environment for each combination of MLIR
    observation/reward/benchmark."""

    register(
        id="mlir-v0",
        entry_point="compiler_gym.envs.mlir:MlirEnv",
        kwargs={
            "service": MLIR_SERVICE_BINARY,
        },
    )

    # TODO(boian): Make better config
    # observation_spaces = {"runtime": "Runtime"}
    # reward_spaces = {"runtime": "Runtime"}
    # for reward_space in reward_spaces:
    #     register(
    #         id=f"mlir-{reward_space}-v0",
    #         entry_point="compiler_gym.envs.mlir:MlirEnv",
    #         kwargs={
    #             "service": MLIR_SERVICE_BINARY,
    #             "reward_space": reward_spaces[reward_space],
    #         },
    #     )
    #
    # for observation_space, reward_space in product(observation_spaces, reward_spaces):
    #     register(
    #         id=f"mlir-{observation_space}-{reward_space}-v0",
    #         entry_point="compiler_gym.envs.mlir:MlirEnv",
    #         kwargs={
    #             "service": MLIR_SERVICE_BINARY,
    #             "observation_space": observation_spaces[observation_space],
    #             "reward_space": reward_spaces[reward_space],
    #         },
    #     )


_register_mlir_gym_service()
