# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from pathlib import Path

from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

EXAMPLE_CC_SERVICE_BINARY: Path = runfiles_path(
    "examples/example_compiler_gym_service/service_cc/compiler_gym-example-service-cc"
)

EXAMPLE_PY_SERVICE_BINARY: Path = runfiles_path(
    "examples/example_compiler_gym_service/service_py/compiler_gym-example-service-py"
)


class RuntimeReward(Reward):
    """An example reward that uses changes in the "runtime" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="runtime",
            observation_spaces=["runtime"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_runtime = None

    def reset(self, benchmark: str):
        del benchmark  # unused
        self.previous_runtime = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_runtime is None:
            self.previous_runtime = observations[0]

        reward = float(self.previous_runtime - observations[0])
        self.previous_runtime = observations[0]
        return reward


# Register the example service on module import. After importing this module,
# the example-v0 environment will be available to gym.make(...).
register(
    id="example-cc-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": EXAMPLE_CC_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
    },
)

register(
    id="example-py-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": EXAMPLE_PY_SERVICE_BINARY,
        "rewards": [RuntimeReward()],
    },
)
