# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module implements a wrapper that counts calls to operations.
"""
from typing import Dict

from compiler_gym.envs import CompilerEnv
from compiler_gym.wrappers import CompilerEnvWrapper


class Counter(CompilerEnvWrapper):
    """A wrapper that counts the number of calls to its operations.

    The counters are _not_ reset by :meth:`env.reset()
    <compiler_gym.envs.CompilerEnv.reset>`.

    Example usage:

        >>> env = Counter(compiler_gym.make("llvm-v0"))
        >>> env.counters
        {"close": 0, "reset": 0, "step": 0, "fork": 0}
        >>> env.step(0)
        {"close": 0, "reset": 0, "step": 1, "fork": 0}

    :ivar counters: A dictionary of counters for different operation types.

    :vartype counters: Dict[str, int]
    """

    def __init__(self, env: CompilerEnv):
        """Constructor.

        :param env: The environment to wrap.
        """
        super().__init__(env)
        self.counters: Dict[str, int] = {
            "close": 0,
            "reset": 0,
            "step": 0,
            "fork": 0,
        }

    def close(self) -> None:
        self.counters["close"] += 1
        self.env.close()

    def reset(self, *args, **kwargs):
        self.counters["reset"] += 1
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.counters["step"] += 1
        return self.env.step(*args, **kwargs)

    def fork(self):
        self.counters["fork"] += 1
        return self.env.fork()
