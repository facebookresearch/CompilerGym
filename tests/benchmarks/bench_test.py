# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Microbenchmarks for CompilerGym environments."""
import gym
import pytest

from compiler_gym.envs import CompilerEnv, LlvmEnv
from compiler_gym.service import CompilerGymServiceConnection
from tests.envs.llvm.fixtures import SERVICE_BIN
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

# Redefine this fixture to run only the local LLVM env.
@pytest.fixture(scope="function")
def env() -> CompilerEnv:
    """Create an LLVM environment."""
    env = gym.make("llvm-v0")
    env.require_dataset("cBench-v0")
    try:
        yield env
    finally:
        env.close()


# Redefine this fixture since running all of the benchmarks in cBench would
# take too long, but we do want to use at least one small and one large
# benchmark to see both per-invocation overhead and overhead that is a result
# of the size of the fixture.
#
# adpcm is small and jpeg-d is large. ghostscript is the largest but that
# one takes too long.
@pytest.fixture(params=["cBench-v0/adpcm", "cBench-v0/jpeg-d"])
def benchmark_name(request) -> str:
    yield request.param


# The observation benchmark is too slow for a large input.
@pytest.fixture(params=["cBench-v0/adpcm"])
def fast_benchmark_name(request) -> str:
    yield request.param


def test_make_local(benchmark):
    benchmark(lambda: gym.make("llvm-v0").close())


def test_make_service(benchmark):
    service = CompilerGymServiceConnection(SERVICE_BIN)
    try:
        benchmark(lambda: LlvmEnv(service=service.connection.url).close())
    finally:
        service.close()


def test_reset(benchmark, env: CompilerEnv, benchmark_name):
    benchmark(env.reset, benchmark_name)


def test_step(benchmark, env: CompilerEnv, benchmark_name):
    env.reset(benchmark_name)
    benchmark(env.step, 0)


def test_observation(
    benchmark, env: CompilerEnv, fast_benchmark_name, observation_space
):
    env.reset(fast_benchmark_name)
    benchmark(lambda: env.observation[observation_space])


def test_reward(benchmark, env: CompilerEnv, benchmark_name, reward_space):
    env.reset(benchmark_name)
    benchmark(lambda: env.reward[reward_space])


if __name__ == "__main__":
    main()
