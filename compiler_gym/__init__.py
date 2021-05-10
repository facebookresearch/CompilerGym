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
try:
    from compiler_gym.util.version import __version__  # isort:skip
except ModuleNotFoundError as e:
    # NOTE(https://github.com/facebookresearch/CompilerGym/issues/76): Handler
    # for a particularly unhelpful error message.
    raise ModuleNotFoundError(
        f"{e}.\nAre you running in the root of the CompilerGym repository?\n"
        "If so, please change to a different directory so that `import "
        "compiler_gym` will work."
    ) from e

from compiler_gym.compiler_env_state import (
    CompilerEnvState,
    CompilerEnvStateReader,
    CompilerEnvStateWriter,
)
from compiler_gym.envs import COMPILER_GYM_ENVS, CompilerEnv
from compiler_gym.random_search import random_search
from compiler_gym.util.debug_util import (
    get_debug_level,
    get_logging_level,
    set_debug_level,
)
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import (
    cache_path,
    site_data_path,
    transient_cache_path,
)
from compiler_gym.validate import validate_states
from compiler_gym.validation_error import ValidationError
from compiler_gym.validation_result import ValidationResult

# The top-level compiler_gym API.
__all__ = [
    "__version__",
    "cache_path",
    "COMPILER_GYM_ENVS",
    "CompilerEnv",
    "CompilerEnvState",
    "CompilerEnvStateWriter",
    "CompilerEnvStateReader",
    "download",
    "get_debug_level",
    "get_logging_level",
    "random_search",
    "set_debug_level",
    "site_data_path",
    "transient_cache_path",
    "validate_states",
    "ValidationError",
    "ValidationResult",
]
