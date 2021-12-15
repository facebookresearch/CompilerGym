# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the loop_tool CompilerGym environment."""

import pytest
from flaky import flaky

import compiler_gym
from tests.test_main import main

backends = ["cpu"]  # TODO swap to lt.backends() when CUDA interaction is fixed

@pytest.mark.parametrize("backend", backends)
def test_basic(backend):
    with compiler_gym.make("loop_tool-v0") as env:
        env.observation_space = "vars"
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{backend}-v0/32"
            ),
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


@pytest.mark.parametrize("backend", backends)
def test_rand(backend):
    with compiler_gym.make("loop_tool-v0") as env:
        env.observation_space = "flops"
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{backend}-v0/128"
            ),
            action_space="simple",
        )
        best = 0
        for i in range(10):
            a = env.action_space.sample()
            o = env.step(a)
            flops = o[0]
            if flops > best:
                best = flops
                print(best)


@pytest.mark.parametrize("backend", backends)
def test_induced_remainder(backend):
    with compiler_gym.make("loop_tool-v0") as env:
        env.observation_space = "vars"
        # reset
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{backend}-v0/32"
            ),
            action_space="simple",
        )
        # action toggle_mode
        o = env.step(0)
        vs = o[0].strip().split(",")
        assert len(vs) == 1
        v = vs[0]
        # action up
        env.step(1)
        # action toggle_mode
        env.step(0)
        # action up
        env.step(1)
        # action up
        env.observation_space = "loop_tree"
        o = env.step(1)
        expected = f"""
for {v} in 10 r 2 : L0 {'parallel ' if backend=='cpu' else ''}[thread]
 for {v}' in 3 : L1
  for {v}'' in 1 : L2
   %0[{v}] <- read()
  for {v}'' in 1 : L4
   %1[{v}] <- read()
  for {v}'' in 1 : L6
   %2[{v}] <- add(%0, %1)
  for {v}'' in 1 : L8
   %3[{v}] <- write(%2)
"""
        lines = o[0].strip().split("\n")
        out = "\n".join(line.rstrip() for line in lines)
        assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


@pytest.mark.parametrize("backend", backends)
def test_thread_removal(backend):
    with compiler_gym.make("loop_tool-v0") as env:
        env.observation_space = "vars"
        # reset
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{backend}-v0/32"
            ),
            action_space="simple",
        )
        # action toggle_thread
        o = env.step(3)
        vs = o[0].strip().split(",")
        assert len(vs) == 1
        v = vs[0]

        # action toggle_thread
        env.step(3)
        # action toggle_thread
        env.observation_space = "loop_tree"
        o = env.step(3)
        expected = f"""
for {v} in 32 : L0
 for {v}' in 1 : L1
  for {v}'' in 1 : L2
   %0[{v}] <- read()
  for {v}'' in 1 : L4
   %1[{v}] <- read()
  for {v}'' in 1 : L6
   %2[{v}] <- add(%0, %1)
  for {v}'' in 1 : L8
   %3[{v}] <- write(%2)
"""
        lines = o[0].strip().split("\n")
        out = "\n".join(line.rstrip() for line in lines)
        assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


@pytest.mark.parametrize("backend", backends)
def test_thread_addition(backend):
    with compiler_gym.make("loop_tool-v0") as env:
        env.observation_space = "vars"
        # reset
        env.reset(
            benchmark=env.datasets.benchmark(
                uri=f"benchmark://loop_tool-{backend}-v0/32"
            ),
            action_space="simple",
        )
        # action toggle_mode
        o = env.step(0)
        vs = o[0].strip().split(",")
        assert len(vs) == 1
        v = vs[0]

        # action up
        env.step(1)
        env.observation_space = "loop_tree"
        # action toggle_thread
        o = env.step(3)
        expected = f"""
for {v} in 32 : L0 {'parallel ' if backend=='cpu' else ''}[thread]
 for {v}' in 1 : L1 {'parallel ' if backend=='cpu' else ''}[thread]
  for {v}'' in 1 : L2
   %0[{v}] <- read()
  for {v}'' in 1 : L4
   %1[{v}] <- read()
  for {v}'' in 1 : L6
   %2[{v}] <- add(%0, %1)
  for {v}'' in 1 : L8
   %3[{v}] <- write(%2)
"""
        lines = o[0].strip().split("\n")
        out = "\n".join(line.rstrip() for line in lines)
        assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


if __name__ == "__main__":
    main()
