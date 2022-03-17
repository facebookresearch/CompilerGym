# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.wrappers.core import CompilerEnvWrapper


class ValidateBenchmarkAfterEveryStep(CompilerEnvWrapper):
    """Run the benchmark validation routine after every step of the environment
    and end the episode with a penalty reward if validation fails.
    """

    def __init__(
        self,
        env: CompilerEnv,
        reward_penalty: float = -1e3,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param reward_penalty: The reward value that is returned by
            :code:`step()` if validation fails.
        """
        super().__init__(env)
        self.reward_penalty = reward_penalty

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        reward_spaces=None,
        observations=None,
        rewards=None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )

        # Early exit if environment reaches terminal state.
        if done:
            return observation, reward, done, info

        try:
            # Try and get an error from the validation callback.
            info["error_details"] = next(self.env.benchmark.ivalidate(self.env))
            return observation, self.reward_penalty, True, info
        except StopIteration:
            # No error, we're good.
            return observation, reward, done, info
