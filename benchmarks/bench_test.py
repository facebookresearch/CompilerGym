# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Microbenchmarks for CompilerGym environments.

To run these benchmarks an optimized build using bazel:

    $ bazel test -c opt --test_output=streamed //benchmarks:bench_test

A record of the benchmark results is stored in
/tmp/compiler_gym/pytest_benchmark/<device>/<run_id>_bench_test.json. Compare
multiple runs using:

    $ pytest-benchmark compare --group-by=name --sort=fullname \
        /tmp/compiler_gym/pytest_benchmark/*/*_bench_test.json
"""
import gym
import pytest

import examples.example_compiler_gym_service as dummy
from compiler_gym.envs import CompilerEnv, LlvmEnv, llvm
from compiler_gym.service import CompilerGymServiceConnection
from tests.pytest_plugins.llvm import OBSERVATION_SPACE_NAMES, REWARD_SPACE_NAMES
from tests.test_main import main


@pytest.fixture(
    params=["llvm-v0", "example-cc-v0", "example-py-v0"],
    ids=["llvm", "dummy-cc", "dummy-py"],
)
def env_id(request) -> str:
    yield request.param


@pytest.fixture(
    params=["llvm-v0", "example-cc-v0", "example-py-v0"],
    ids=["llvm", "dummy-cc", "dummy-py"],
)
def env(request) -> CompilerEnv:
    yield request.param


@pytest.mark.parametrize(
    "env_id",
    ["llvm-v0", "example-cc-v0", "example-py-v0"],
    ids=["llvm", "dummy-cc", "dummy-py"],
)
def test_make_local(benchmark, env_id):
    benchmark(lambda: gym.make(env_id).close())


@pytest.mark.parametrize(
    "args",
    [
        (llvm.LLVM_SERVICE_BINARY, LlvmEnv),
        (dummy.EXAMPLE_CC_SERVICE_BINARY, CompilerEnv),
        (dummy.EXAMPLE_PY_SERVICE_BINARY, CompilerEnv),
    ],
    ids=["llvm", "dummy-cc", "dummy-py"],
)
def test_make_service(benchmark, args):
    service_binary, env_class = args
    service = CompilerGymServiceConnection(service_binary)
    try:
        benchmark(lambda: env_class(service=service.connection.url).close())
    finally:
        service.close()


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/crc32"),
        lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/jpeg-d"),
        lambda: gym.make("example-cc-v0"),
        lambda: gym.make("example-py-v0"),
    ],
    ids=["llvm;fast-benchmark", "llvm;slow-benchmark", "dummy-cc", "dummy-py"],
)
def test_reset(benchmark, make_env: CompilerEnv):
    with make_env() as env:
        benchmark(env.reset)


@pytest.mark.parametrize(
    "args",
    [
        (
            lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/crc32"),
            "-globaldce",
        ),
        (lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/crc32"), "-gvn"),
        (
            lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/jpeg-d"),
            "-globaldce",
        ),
        (
            lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/jpeg-d"),
            "-gvn",
        ),
        (lambda: gym.make("example-cc-v0"), "a"),
        (lambda: gym.make("example-py-v0"), "a"),
    ],
    ids=[
        "llvm;fast-benchmark;fast-action",
        "llvm;fast-benchmark;slow-action",
        "llvm;slow-benchmark;fast-action",
        "llvm;slow-benchmark;slow-action",
        "dummy-cc",
        "dummy-py",
    ],
)
def test_step(benchmark, args):
    make_env, action_name = args
    with make_env() as env:
        env.reset()
        action = env.action_space[action_name]
        benchmark(env.step, action)


_args = dict(
    {
        f"llvm;{obs}": (lambda: gym.make("llvm-v0", benchmark="cbench-v1/qsort"), obs)
        for obs in OBSERVATION_SPACE_NAMES
    },
    **{
        "dummy-cc": (lambda: gym.make("example-cc-v0"), "ir"),
        "dummy-py": (lambda: gym.make("example-py-v0"), "features"),
    },
)


@pytest.mark.parametrize("args", _args.values(), ids=_args.keys())
def test_observation(benchmark, args):
    make_env, observation_space = args
    with make_env() as env:
        env.reset()
        benchmark(lambda: env.observation[observation_space])


_args = dict(
    {
        f"llvm;{reward}": (
            lambda: gym.make("llvm-v0", benchmark="cbench-v1/qsort"),
            reward,
        )
        for reward in REWARD_SPACE_NAMES
    },
    **{
        "dummy-cc": (lambda: gym.make("example-cc-v0"), "runtime"),
        "dummy-py": (lambda: gym.make("example-py-v0"), "runtime"),
    },
)


@pytest.mark.parametrize("args", _args.values(), ids=_args.keys())
def test_reward(benchmark, args):
    make_env, reward_space = args
    with make_env() as env:
        env.reset()
        benchmark(lambda: env.reward[reward_space])


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/crc32"),
        lambda: gym.make("llvm-autophase-ic-v0", benchmark="cbench-v1/jpeg-d"),
        # TODO: Example service does not yet support fork() operator.
        # lambda: gym.make("example-cc-v0"),
        # lambda: gym.make("example-py-v0"),
    ],
    ids=["llvm;fast-benchmark", "llvm;slow-benchmark"],
)
def test_fork(benchmark, make_env):
    with make_env() as env:
        env.reset()
        benchmark(lambda: env.fork().close())


if __name__ == "__main__":
    main(
        extra_pytest_args=[
            "--benchmark-storage=/tmp/compiler_gym/pytest_benchmark",
            "--benchmark-save=bench_test",
            "--benchmark-sort=name",
            "-x",
        ],
        debug_level=0,
    )
