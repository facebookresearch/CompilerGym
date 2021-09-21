# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the loop_tool CompilerGym environment."""

import pytest

import compiler_gym
from tests.test_main import main


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_basic(backend):
    env = compiler_gym.make("looptool-v0")
    env.observation_space = "flops"
    env.reset(
        benchmark=env.datasets.benchmark(
            uri=f"benchmark://loop_tool-{backend}-v0/1024"
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


@pytest.mark.parametrize("backend", ["cuda"])
def test_rand(backend):
    env = compiler_gym.make("looptool-v0")
    env.observation_space = "flops"
    env.reset(
        benchmark=env.datasets.benchmark(
            uri=f"benchmark://loop_tool-{backend}-v0/1024"
        ),
        action_space="simple",
    )
    best = 0
    for i in range(20):
        a = env.action_space.sample()
        o = env.step(a)
        flops = o[0]
        if flops > best:
            best = flops
            print(best)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_induced_remainder(backend):
    env = compiler_gym.make("looptool-v0")
    env.observation_space = "loop_tree"
    # reset
    env.reset(
        benchmark=env.datasets.benchmark(
            uri=f"benchmark://loop_tool-{backend}-v0/1024"
        ),
        action_space="simple",
    )
    # action toggle_mode
    env.step(0)
    # action up
    env.step(1)
    # action toggle_mode
    env.step(0)
    # action up
    env.step(1)
    # action up
    o = env.step(1)
    expected = f"""
for a in 341 r 1 : L0 {'cpu_parallel ' if backend=='cpu' else ''}[thread]
 for a' in 3 : L1
  for a'' in 1 : L2
   %0[a] <- read()
  for a'' in 1 : L4
   %1[a] <- read()
  for a'' in 1 : L6
   %2[a] <- add(%0, %1)
  for a'' in 1 : L8
   %3[a] <- write(%2)
"""
    lines = o[0].strip().split("\n")
    out = "\n".join(line.rstrip() for line in lines)
    assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_thread_removal(backend):
    env = compiler_gym.make("looptool-v0")
    env.observation_space = "loop_tree"
    # reset
    env.reset(
        benchmark=env.datasets.benchmark(
            uri=f"benchmark://loop_tool-{backend}-v0/1024"
        ),
        action_space="simple",
    )
    # action toggle_thread
    o = env.step(3)
    expected = """
for a in 1024 : L0
 for a' in 1 : L1
  for a'' in 1 : L2
   %0[a] <- read()
  for a'' in 1 : L4
   %1[a] <- read()
  for a'' in 1 : L6
   %2[a] <- add(%0, %1)
  for a'' in 1 : L8
   %3[a] <- write(%2)
"""
    lines = o[0].strip().split("\n")
    out = "\n".join(line.rstrip() for line in lines)
    assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_thread_addition(backend):
    env = compiler_gym.make("looptool-v0")
    env.observation_space = "loop_tree"
    # reset
    env.reset(
        benchmark=env.datasets.benchmark(
            uri=f"benchmark://loop_tool-{backend}-v0/1024"
        ),
        action_space="simple",
    )
    # action toggle_mode
    env.step(0)
    # action up
    env.step(1)
    # action toggle_thread
    o = env.step(3)
    expected = f"""
for a in 1024 : L0 {'cpu_parallel ' if backend=='cpu' else ''}[thread]
 for a' in 1 : L1 {'cpu_parallel ' if backend=='cpu' else ''}[thread]
  for a'' in 1 : L2
   %0[a] <- read()
  for a'' in 1 : L4
   %1[a] <- read()
  for a'' in 1 : L6
   %2[a] <- add(%0, %1)
  for a'' in 1 : L8
   %3[a] <- write(%2)
"""
    lines = o[0].strip().split("\n")
    out = "\n".join(line.rstrip() for line in lines)
    assert out == expected.strip(), f"{out} \n vs \n {expected.strip()}"


if __name__ == "__main__":
    main(debug_level=4)
