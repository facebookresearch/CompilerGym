# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from itertools import product

from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

_EXAMPLE_SERVICE_BINARY = runfiles_path(
    "CompilerGym/examples/example_compiler_gym_service/service/service"
)


def _register_example_gym_service():
    """Register environments for the example service."""
    register(
        id="example-v0",
        entry_point="compiler_gym.envs:CompilerEnv",
        kwargs={
            "service": _EXAMPLE_SERVICE_BINARY,
        },
    )

    # Register rewards for all combinations of eager observation and
    # reward spaces.
    observation_spaces = ["ir", "features"]
    reward_spaces = ["codesize"]
    configurations = product(observation_spaces, reward_spaces)
    for observation_space, reward_space in configurations:
        env_id = f"example-{observation_space}-{reward_space}-v0"
        register(
            id=env_id,
            entry_point="compiler_gym.envs:CompilerEnv",
            kwargs={
                "service": _EXAMPLE_SERVICE_BINARY,
                "observation_space": observation_space,
                "reward_space": reward_space,
            },
        )


_register_example_gym_service()
