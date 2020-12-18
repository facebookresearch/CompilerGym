# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from gym.envs.registration import register as gym_register

# A list of gym environment names defined by CompilerGym.
COMPILER_GYM_ENVS: List[str] = []


def register(id: str, **kwargs):
    COMPILER_GYM_ENVS.append(id)
    gym_register(id=id, **kwargs)
