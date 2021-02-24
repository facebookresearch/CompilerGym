# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Microbenchmarks for CompilerGym environments.

To run these benchmarks within bazel, compile with optimiztions and stream the
test output:

    $ bazel test -c opt --test_output=streamed //tests/benchmarks:bench_test

A record of the benchmark results is stored in
/tmp/compiler_gym/benchmarks/<device>/<run>_bench_test.json
Compare multiple runs using:

    $ pytest-benchmark compare --group-by=name --sort=fullname \
        /tests/benchmarks/*_bench_test.json
"""
import gym
import pytest

from compiler_gym.envs import CompilerEnv, LlvmEnv, llvm
from compiler_gym.service import CompilerGymServiceConnection
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# Redefine this fixture since running all of the benchmarks in cBench would
# take too long, but we do want to use at least one small and one large
# benchmark to see both per-invocation overhead and overhead that is a result
# of the size of the fixture.
#
# adpcm is small and jpeg-d is large. ghostscript is the largest but that
# one takes too long.
@pytest.fixture(
    params=["cBench-v0/crc32", "cBench-v0/jpeg-d"],
    ids=["fast_benchmark", "slow_benchmark"],
)
def benchmark_name(request) -> str:
    yield request.param


@pytest.fixture(params=["cBench-v0/crc32"], ids=["fast_benchmark"])
def fast_benchmark_name(request) -> str:
    yield request.param


@pytest.fixture(params=["-globaldce", "-gvn"], ids=["fast_action", "slow_action"])
def action_name(request) -> str:
    yield request.param


def test_make_local(benchmark):
    benchmark(lambda: gym.make("llvm-v0").close())


def test_make_service(benchmark):
    service = CompilerGymServiceConnection(llvm.LLVM_SERVICE_BINARY)
    try:
        benchmark(lambda: LlvmEnv(service=service.connection.url).close())
    finally:
        service.close()


def test_reset(benchmark, env: CompilerEnv, benchmark_name):
    benchmark(env.reset, benchmark_name)


def test_step(benchmark, env: CompilerEnv, benchmark_name, action_name):
    env.reset(benchmark_name)
    action = env.action_space.flags.index(action_name)
    benchmark(env.step, action)


def test_observation(
    benchmark, env: CompilerEnv, fast_benchmark_name, observation_space
):
    env.reset(fast_benchmark_name)
    benchmark(lambda: env.observation[observation_space])


def test_reward(benchmark, env: CompilerEnv, benchmark_name, reward_space):
    env.reset(benchmark_name)
    benchmark(lambda: env.reward[reward_space])


def test_fork(benchmark, env: CompilerEnv, benchmark_name):
    env.reset(benchmark_name)
    benchmark(lambda: env.fork().close())


if __name__ == "__main__":
    main(
        extra_pytest_args=[
            "--benchmark-storage=/tmp/compiler_gym/benchmarks",
            "--benchmark-save=bench_test",
            "-x",
        ],
        service_debug=0,
    )
