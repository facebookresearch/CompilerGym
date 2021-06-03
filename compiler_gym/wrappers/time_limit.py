# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import gym

from compiler_gym.wrappers.core import CompilerEnvWrapper


class TimeLimit(gym.wrappers.TimeLimit, CompilerEnvWrapper):
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
