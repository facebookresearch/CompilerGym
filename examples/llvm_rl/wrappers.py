# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Environment wrappers to closer replicate the MLSys'20 Autophase paper."""
from typing import List

import gym
import numpy as np

from compiler_gym.envs import CompilerEnv, LlvmEnv
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.wrappers import (
    ConstrainedCommandline,
    ObservationWrapper,
    RewardWrapper,
)


class ClampedReward(RewardWrapper):
    """A wrapper class that clamps reward signal within a bounded range,
    optionally with some leaking for out-of-range values.
    """

    def __init__(
        self,
        env: CompilerEnv,
        min: float = -1,
        max: float = 1,
        leakiness_factor: float = 0.001,
    ):
        super().__init__(env)
        self.min = min
        self.max = max
        self.leakiness_factor = leakiness_factor

    def reward(self, reward: float) -> float:
        if reward > self.max:
            return self.max + (reward - self.max) * self.leakiness_factor
        elif reward < self.min:
            return self.min + (reward - self.min) * self.leakiness_factor
        return reward


class AutophaseNormalizedFeatures(ObservationWrapper):
    """A wrapper for LLVM environments that use the Autophase observation space
    to normalize and clip features to the range [0, 1].
    """

    # The index of the "TotalInsts" feature of autophase.
    TotalInsts_index = 51

    def __init__(self, env: CompilerEnv):
        super().__init__(env=env)
        # Force Autophase observation space.
        self.env.observation_space = self.env.unwrapped.observation.spaces["Autophase"]
        # Adjust the bounds to reflect the normalized values.
        self.env.observation_space_spec.space = gym.spaces.Box(
            low=np.full(
                self.env.observation_space_spec.space.shape[0], 0, dtype=np.float32
            ),
            high=np.full(
                self.env.observation_space_spec.space.shape[0], 1, dtype=np.float32
            ),
            dtype=np.float32,
        )

    def observation(self, observation):
        if observation[self.TotalInsts_index] <= 0:
            return np.zeros(observation.shape, dtype=np.float32)
        return np.clip(
            observation.astype(np.float32) / observation[self.TotalInsts_index], 0, 1
        )


class ConcatActionsHistogram(ObservationWrapper):
    """A wrapper that concatenates a histogram of previous actions to each
    observation.

    The actions histogram is concatenated to the end of the existing 1-D box
    observation, expanding the space.

    The actions histogram has bounds [0,inf]. If you specify a fixed episode
    length `norm_to_episode_len`, each histogram update will be scaled by
    1/norm_to_episode_len, so that `sum(observation) == 1` after episode_len
    steps.
    """

    def __init__(self, env: CompilerEnv, norm_to_episode_len: int = 0):
        super().__init__(env=env)
        assert isinstance(
            self.observation_space, gym.spaces.Box
        ), f"Can only contatenate actions histogram to box shape, not {self.observation_space}"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Can only construct histograms from discrete spaces"
        assert len(self.observation_space.shape) == 1, "Requires 1-D observation space"
        self.increment = 1 / norm_to_episode_len if norm_to_episode_len else 1

        # Reshape the observation space.
        self.env.observation_space_spec.space = gym.spaces.Box(
            low=np.concatenate(
                (
                    self.env.observation_space.low,
                    np.zeros(
                        self.action_space.n, dtype=self.env.observation_space.dtype
                    ),
                )
            ),
            high=np.concatenate(
                (
                    self.env.observation_space.high,
                    # The upper bound is 1.0 if we are normalizing to the
                    # episode length, else infinite for unbounded episode
                    # lengths.
                    np.ones(self.action_space.n, dtype=self.env.observation_space.dtype)
                    * (1.0 if norm_to_episode_len else np.inf),
                )
            ),
            dtype=self.env.observation_space.dtype,
        )

    def reset(self, *args, **kwargs):
        self.histogram = np.zeros(
            (self.action_space.n,), dtype=self.env.observation_space.dtype
        )
        return super().reset(*args, **kwargs)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        observations=None,
        **kwargs,
    ):
        for a in actions:
            self.histogram[a] += self.increment
        return super().multistep(actions, **kwargs)

    def observation(self, observation):
        return np.concatenate((observation, self.histogram)).astype(
            self.env.observation_space.dtype
        )


class AutophaseActionSpace(ConstrainedCommandline):
    """An action space wrapper that limits the action space to that of the
    Autophase paper.

    The actions used in the Autophase work are taken from:

    https://github.com/ucb-bar/autophase/blob/2f2e61ad63b50b5d0e2526c915d54063efdc2b92/gym-hls/gym_hls/envs/getcycle.py#L9

    Note that 4 of the 46 flags are not included. Those are:

        -codegenprepare     Excluded from CompilerGym
            -scalarrepl     Removed from LLVM in https://reviews.llvm.org/D21316
        -scalarrepl-ssa     Removed from LLVM in https://reviews.llvm.org/D21316
             -terminate     Not found in LLVM 10.0.0
    """

    def __init__(self, env: LlvmEnv):
        super().__init__(
            env=env,
            flags=[
                "-adce",
                "-break-crit-edges",
                "-constmerge",
                "-correlated-propagation",
                "-deadargelim",
                "-dse",
                "-early-cse",
                "-functionattrs",
                "-functionattrs",
                "-globaldce",
                "-globalopt",
                "-gvn",
                "-indvars",
                "-inline",
                "-instcombine",
                "-ipsccp",
                "-jump-threading",
                "-lcssa",
                "-licm",
                "-loop-deletion",
                "-loop-idiom",
                "-loop-reduce",
                "-loop-rotate",
                "-loop-simplify",
                "-loop-unroll",
                "-loop-unswitch",
                "-lower-expect",
                "-loweratomic",
                "-lowerinvoke",
                "-lowerswitch",
                "-mem2reg",
                "-memcpyopt",
                "-partial-inliner",
                "-prune-eh",
                "-reassociate",
                "-sccp",
                "-simplifycfg",
                "-sink",
                "-sroa",
                "-strip",
                "-strip-nondebug",
                "-tailcallelim",
            ],
        )
