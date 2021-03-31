# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generate enum declarations for LLVM service capabilities.

Usage: make_specs.py <service_binary> <output_path>.
"""
# TODO: As we add support for more compilers we could generalize this script
# to work with other compiler services rather than hardcoding to LLVM.
import sys
from pathlib import Path

from compiler_gym.envs.llvm.llvm_env import LlvmEnv


def main(argv):
    assert len(argv) == 3, "Usage: make_specs.py <service_binary> <output_path>"
    service_path, output_path = argv[1:]

    env = LlvmEnv(Path(service_path))
    try:
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
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv)
