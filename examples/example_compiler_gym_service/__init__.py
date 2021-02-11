# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path

# Register the example service on module import. After importing this module,
# the example-v0 environment will be available to gym.make(...).
register(
    id="example-v0",
    entry_point="compiler_gym.envs:CompilerEnv",
    kwargs={
        "service": runfiles_path(
            "examples/example_compiler_gym_service/service/compiler_gym-example-service"
        ),
    },
)
