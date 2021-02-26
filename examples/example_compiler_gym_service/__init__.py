# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path


class CodesizeReward(Reward):
    """An example reward that uses changes in the "codesize" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="codesize",
            observation_spaces=["codesize"],
            default_value=0,
            default_negates_returns=True,
            deterministic=True,
            platform_dependent=False,
        )
        self.previous_codesize = None

    def reset(self, benchmark: str):
        del benchmark  # unused
        self.previous_codesize = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_codesize is None:
            self.previous_codesize = observations[0]
        reward = float(self.previous_codesize - observations[0])
        self.previous_codesize = observations[0]
        return reward


# Register the example service on module import. After importing this module,
# the example-v0 environment will be available to gym.make(...).
register(
    id="example-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": runfiles_path(
            "examples/example_compiler_gym_service/service/compiler_gym-example-service"
        ),
        "rewards": [CodesizeReward()],
    },
)
