# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CompilerGym is a set of compiler optimization environments for reinforcement learning.

After importing this module, the :class:`CompilerGym environments <compiler_gym.envs.CompilerEnv>`
will be available through the :code:`gym.make(...)` interface:

    >>> import gym
    >>> import compiler_gym
    >>> gym.make("llvm-v0")

The list of CompilerGym environments that can be passed to :code:`gym.make(...)`
is available through :code:`compiler_gym.COMPILER_GYM_ENVS`:

    >>> import compiler_gym
    >>> compiler_gym.COMPILER_GYM_ENVS
    ['llvm-v0', 'llvm-ic-v0', 'llvm-autophase-ic-v0', 'llvm-ir-ic-v0']
"""
from compiler_gym.util.version import __version__  # isort:skip

from compiler_gym.envs import COMPILER_GYM_ENVS, CompilerEnv, observation_t, step_t
from compiler_gym.random_search import random_search
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path
from compiler_gym.validate import ValidationResult, validate_state, validate_states

# The top-level compiler_gym API.
__all__ = [
    "__version__",
    "download",
    "cache_path",
    "site_data_path",
    "CompilerEnv",
    "COMPILER_GYM_ENVS",
    "observation_t",
    "step_t",
    "random_search",
    "ValidationResult",
    "validate_state",
    "validate_states",
]
