# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import gym
from gym.envs.registration import register as gym_register

# A list of gym environment names defined by CompilerGym.
COMPILER_GYM_ENVS: List[str] = []


def make(id: str, **kwargs):
    """Equivalent to :code:`gym.make()`."""
    return gym.make(id, **kwargs)


def _parse_version_string(version):
    """Quick and dirty <major>.<minor>.<micro> parser. Very hacky."""
    components = version.split(".")
    if len(components) != 3:
        return None
    try:
        return tuple([int(x) for x in components])
    except (TypeError, ValueError):
        return None


def register(id: str, order_enforce: bool = False, **kwargs):
    COMPILER_GYM_ENVS.append(id)

    # As of gym==0.21.0 a new OrderEnforcing wrapper is enabled by default. Turn
    # this off as CompilerEnv already enforces this and the wrapper obscures the
    # docstrings of the base class.
    gym_version = _parse_version_string(gym.__version__)
    if gym_version and gym_version >= (0, 21):
        kwargs["order_enforce"] = order_enforce

    gym_register(id=id, **kwargs)
