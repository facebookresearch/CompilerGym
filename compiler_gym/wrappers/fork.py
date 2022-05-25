# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module implements fork wrappers."""
from typing import List

from compiler_gym.envs import CompilerEnv
from compiler_gym.wrappers import CompilerEnvWrapper


class ForkOnStep(CompilerEnvWrapper):
    """A wrapper that creates a fork of the environment before every step.

    This wrapper creates a new fork of the environment before every call to
    :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>`. Because of this,
    this environment supports an additional :meth:`env.undo()
    <compiler_gym.wrappers.ForkOnStep.undo>` method that can be used to
    backtrack.

    Example usage:

        >>> env = ForkOnStep(compiler_gym.make("llvm-v0"))
        >>> env.step(0)
        >>> env.actions
        [0]
        >>> env.undo()
        >>> env.actions
        []

    :ivar stack: A fork of the environment before every previous call to
        :meth:`env.reset() <compiler_gym.envs.CompilerEnv.reset>`, ordered
        oldest to newest.

    :vartype stack: List[CompilerEnv]
    """

    def __init__(self, env: CompilerEnv):
        """Constructor.

        :param env: The environment to wrap.
        """
        super().__init__(env)
        self.stack: List[CompilerEnv] = []

    def undo(self) -> CompilerEnv:
        """Undo the previous action.

        :returns: Self.
        """
        if not self.stack:
            return
        self.env.close()
        self.env = self.stack.pop()
        return self.env

    def close(self) -> None:
        for env in self.stack:
            env.close()
        self.stack: List[CompilerEnv] = []
        self.env.close()
        self.custom_close = True

    def reset(self, *args, **kwargs):
        self.env.reset()
        for env in self.stack:
            env.close()
        self.stack: List[CompilerEnv] = []

    def step(self, *args, **kwargs):
        self.stack.append(self.env.fork())
        return self.env.step(*args, **kwargs)

    def fork(self):
        raise NotImplementedError
