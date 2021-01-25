# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Register the LLVM environments."""
from itertools import product

from compiler_gym.envs.llvm.benchmarks import (
    ClangInvocation,
    get_system_includes,
    make_benchmark,
)
from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

__all__ = [
    "LlvmEnv",
    "make_benchmark",
    "ClangInvocation",
    "get_system_includes",
    "LLVM_SERVICE_BINARY",
]

LLVM_SERVICE_BINARY = runfiles_path(
    "CompilerGym/compiler_gym/envs/llvm/service/service"
)


def _register_llvm_gym_service():
    """Register an environment for each combination of LLVM
    observation/reward/benchmark."""
    observation_spaces = {"autophase": "Autophase", "ir": "Ir"}
    reward_spaces = {"ic": "IrInstructionCountOz", "codesize": "ObjectTextSizeOz"}

    register(
        id="llvm-v0",
        entry_point="compiler_gym.envs.llvm:LlvmEnv",
        kwargs={
            "service": LLVM_SERVICE_BINARY,
        },
    )

    for reward_space in reward_spaces:
        register(
            id=f"llvm-{reward_space}-v0",
            entry_point="compiler_gym.envs.llvm:LlvmEnv",
            kwargs={
                "service": LLVM_SERVICE_BINARY,
                "reward_space": reward_spaces[reward_space],
            },
        )

    for observation_space, reward_space in product(observation_spaces, reward_spaces):
        register(
            id=f"llvm-{observation_space}-{reward_space}-v0",
            entry_point="compiler_gym.envs.llvm:LlvmEnv",
            kwargs={
                "service": LLVM_SERVICE_BINARY,
                "observation_space": observation_spaces[observation_space],
                "reward_space": reward_spaces[reward_space],
            },
        )


_register_llvm_gym_service()
