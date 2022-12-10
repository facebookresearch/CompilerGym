# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from pathlib import Path

from compiler_gym.envs.cgra.DFG import DFG
from compiler_gym.envs.cgra.service.cgra_service import Schedule, CGRA
from compiler_gym.envs.cgra.service.cgra_env import CgraEnv
from compiler_gym.envs.cgra.service.relative_cgra_env import RelativeCgraEnv
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

CGRA_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/cgra/service/compiler_gym-cgra-service"
)
RELATIVE_CGRA_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/cgra/service/compiler_gym-relative-placement-cgra-service"
)

register(
    id="relative-cgra-v0",
    entry_point="compiler_gym.envs.cgra:RelativeCgraEnv",
    kwargs={ "service": RELATIVE_CGRA_SERVICE_BINARY },
)

register(
    id="cgra-v0",
    entry_point="compiler_gym.envs.cgra:CgraEnv",
    kwargs={"service": CGRA_SERVICE_BINARY},
)

__all__ = ["CgraEnv", "DFG", "CGRA", "Schedule", "RelativeCgraEnv"]
