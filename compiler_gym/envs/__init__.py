# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym import config
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.gcc import GccEnv

if config.enable_llvm_env:
    from compiler_gym.envs.llvm.llvm_env import LlvmEnv  # noqa: F401
if config.enable_mlir_env:
    from compiler_gym.envs.mlir.mlir_env import MlirEnv  # noqa: F401

from compiler_gym.envs.loop_tool.loop_tool_env import LoopToolEnv
from compiler_gym.util.registration import COMPILER_GYM_ENVS

__all__ = [
    "COMPILER_GYM_ENVS",
    "CompilerEnv",
    "GccEnv",
    "LoopToolEnv",
]

if config.enable_llvm_env:
    __all__.append("LlvmEnv")
if config.enable_mlir_env:
    __all__.append("MlirEnv")
