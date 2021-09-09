# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from pathlib import Path

from compiler_gym.envs.gcc.gcc import Gcc, GccSpec, Option
from compiler_gym.envs.gcc.gcc_env import DEFAULT_GCC, GccEnv
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

GCC_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/gcc/service/compiler_gym-gcc-service"
)

register(
    id="gcc-v0",
    entry_point="compiler_gym.envs.gcc:GccEnv",
    kwargs={"service": GCC_SERVICE_BINARY},
)

__all__ = ["GccEnv", "GccSpec", "Gcc", "Option", "DEFAULT_GCC"]
