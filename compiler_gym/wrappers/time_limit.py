# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Optional

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.wrappers.core import CompilerEnvWrapper


class TimeLimit(CompilerEnvWrapper):
    """A step-limited wrapper that is compatible with CompilerGym.

    Example usage:

        >>> env = TimeLimit(env, max_episode_steps=3)
        >>> env.reset()
        >>> _, _, done, _ = env.step(0)
        >>> _, _, done, _ = env.step(0)
        >>> _, _, done, _ = env.step(0)
        >>> done
        True
    """

    def __init__(self, env: CompilerEnv, max_episode_steps: Optional[int] = None):
        super().__init__(env=env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def multistep(self, actions: Iterable[ActionType], **kwargs):
        actions = list(actions)
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.multistep(actions, **kwargs)
        self._elapsed_steps += len(actions)
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def fork(self) -> "TimeLimit":
        """Fork the wrapped environment.

        The time limit state of the forked environment is the same as the source
        state.
        """
        fkd = type(self)(env=self.env.fork(), max_episode_steps=self._max_episode_steps)
        fkd._elapsed_steps = self._elapsed_steps  # pylint: disable=protected-access
        return fkd
