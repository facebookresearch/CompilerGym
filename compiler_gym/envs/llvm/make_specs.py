# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generate specifications for the LLVM service capabilities.

Usage: make_specs.py <service_binary> <output_path>.
"""
# TODO: As we add support for more compilers we could generalize this script
# to work with other compiler services rather than hardcoding to LLVM.
import sys
from pathlib import Path

from compiler_gym.envs.llvm.llvm_env import LlvmEnv
from compiler_gym.util.runfiles_path import runfiles_path

with open(
    runfiles_path("compiler_gym/envs/llvm/service/passes/flag_descriptions.txt")
) as f:
    _FLAG_DESCRIPTIONS = [ln.rstrip() for ln in f.readlines()]


def main(argv):
    assert len(argv) == 3, "Usage: make_specs.py <service_binary> <output_path>"
    service_path, output_path = argv[1:]

    with LlvmEnv(Path(service_path)) as env:
        with open(output_path, "w") as f:
            print("from enum import Enum", file=f)
            print(file=f)
            print("class observation_spaces(Enum):", file=f)
            for name in env.observation.spaces:
                print(f'    {name} = "{name}"', file=f)
            print(file=f)
            print("class reward_spaces(Enum):", file=f)
            for name in env.reward.spaces:
                print(f'    {name} = "{name}"', file=f)
            print(file=f)
            print("class actions(Enum):", file=f)
            for name in env.action_space.names:
                enum_name = "".join([x.capitalize() for x in name[1:].split("-")])
                print(f'    {enum_name} = "{name}"', file=f)
            print(file=f)
            print("class action_descriptions(Enum):", file=f)
            for name, description in zip(env.action_space.names, _FLAG_DESCRIPTIONS):
                enum_name = "".join([x.capitalize() for x in name[1:].split("-")])
                sanitized_description = description.replace('" "', "")
                sanitized_description = sanitized_description.replace('"', "")
                print(f'    {enum_name} = "{sanitized_description}"', file=f)


if __name__ == "__main__":
    main(sys.argv)
