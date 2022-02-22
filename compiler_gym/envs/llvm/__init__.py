# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Register the LLVM environments."""
import sys
from itertools import product

from compiler_gym.envs.llvm.compute_observation import compute_observation
from compiler_gym.envs.llvm.llvm_benchmark import (
    ClangInvocation,
    get_system_library_flags,
    make_benchmark,
)
from compiler_gym.envs.llvm.llvm_env import LlvmEnv

# TODO(github.com/facebookresearch/CompilerGym/issues/506): Tidy up.
if "compiler_gym.envs.llvm.is_making_specs" not in sys.modules:
    from compiler_gym.envs.llvm.specs import observation_spaces, reward_spaces

from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

__all__ = [
    "ClangInvocation",
    "compute_observation",
    "get_system_library_flags",
    "LLVM_SERVICE_BINARY",
    "LlvmEnv",
    "make_benchmark",
    "observation_spaces",
    "reward_spaces",
]

LLVM_SERVICE_BINARY = runfiles_path(
    "compiler_gym/envs/llvm/service/compiler_gym-llvm-service"
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
