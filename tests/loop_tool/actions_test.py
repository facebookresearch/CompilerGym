# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the loop_tool CompilerGym environment."""

import gym

from tests.test_main import main


def test_basic():
    env = gym.make("looptool-v0")
    env.observation_space = "flops"
    env.reset(
        benchmark=env.datasets.benchmark(uri="benchmark://loop_tool-v0/1024"),
        action_space="simple",
    )
    env.step(0)
    env.step(1)
    env.step(0)
    env.step(1)
    env.step(1)
    env.step(0)
    env.step(1)
    env.step(0)
    o = env.step(1)
    print(o)


def test_rand():
    env = gym.make("looptool-v0")
    env.observation_space = "flops"
    env.reset(
        benchmark=env.datasets.benchmark(uri="benchmark://loop_tool-v0/1024"),
        action_space="simple",
    )
    best = 0
    for i in range(20):
        a = env.action_space.sample()
        o = env.step(a)
        flops = o[0]
        # print(flops)
        if flops > best:
            best = flops
            print(best)


if __name__ == "__main__":
    main()
