# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the example services defined in this directory
can be used as gym environments. Usage:

    $ bazel run -c opt //examples/example_unrolling_service:example
"""
import compiler_gym
import examples.example_unrolling_service as unrolling_service  # noqa Register environments.

with compiler_gym.make(
    "unrolling-py-v0",
    benchmark="unrolling-v0/offsets1",
    observation_space="features",
    reward_space="runtime",
) as env:
    compiler_gym.set_debug_level(4)  # TODO: check why this has no effect

    observation = env.reset()
    print("observation: ", observation)

    print()

    observation, reward, done, info = env.step(env.action_space.sample())
    print("observation: ", observation)
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)

    print()

    observation, reward, done, info = env.step(env.action_space.sample())
    print("observation: ", observation)
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)

    env.reset()

    # TODO: implement write_bitcode(..) or write_ir(..)
    # env.write_bitcode("/tmp/output.bc")
