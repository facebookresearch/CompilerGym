# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the example services defined in this directory
can be used as gym environments. Usage:

    $ bazel run -c opt //examples/example_compiler_gym_service:demo
"""
import logging

import gym

# To use the example services we simply need to import the module which
# registers the environments.
import examples.example_compiler_gym_service  # noqa Register environments


def main():
    # Use debug verbosity to print out extra logging information.
    logging.basicConfig(level=logging.DEBUG)

    # Create the environment using the regular gym.make(...) interface. We could
    # use either the C++ service "example-cc-v0" or the Python service
    # "example-py-v0".
    with gym.make("example-cc-v0") as env:
        env.reset()
        for _ in range(20):
            observation, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()


if __name__ == "__main__":
    main()
