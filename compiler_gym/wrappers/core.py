# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Union

import gym

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ObservationType, StepType


class CompilerEnvWrapper(gym.Wrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow a modular transformation.

    This class is the base class for all wrappers. This class must be used
    rather than :code:`gym.Wrapper` to support the CompilerGym API extensions
    such as the :code:`fork()` method.
    """

    def __init__(self, env: CompilerEnv):
        """Constructor.

        :param env: The environment to wrap.

        :raises TypeError: If :code:`env` is not a :class:`CompilerEnv
            <compiler_gym.envs.CompilerEnv>`.
        """
        super().__init__(env)

    def reset(self, *args, **kwargs) -> ObservationType:
        return self.env.reset(*args, **kwargs)

    def fork(self) -> CompilerEnv:
        return type(self)(env=self.env.fork())


class ActionWrapper(CompilerEnvWrapper):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an action space transformation.
    """

    def step(self, action: Union[int, Iterable[int]]) -> StepType:
        return self.env.step(self.action(action))

    def action(self, action):
        """Translate the action to the new space."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Translate an action from the new space to the wrapped space."""
        raise NotImplementedError
