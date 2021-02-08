# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the LLVM CompilerGym environments."""
import os
from pathlib import Path
from typing import Iterable, List

import gym
import pytest

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.runfiles_path import runfiles_path

ACTIONS_LIST = Path(
    runfiles_path("CompilerGym/compiler_gym/envs/llvm/service/passes/actions_flags.txt")
)

BENCHMARKS_LIST = Path(
    runfiles_path("CompilerGym/compiler_gym/third_party/cBench/benchmarks.txt")
)


def _read_list_file(path: Path) -> Iterable[str]:
    with open(str(path)) as f:
        for action in f:
            if action.strip():
                yield action.strip()


ACTION_NAMES = list(_read_list_file(ACTIONS_LIST))
BENCHMARK_NAMES = list(_read_list_file(BENCHMARKS_LIST))

# Skip ghostscript on CI as it is just too heavy.
if bool(os.environ.get("CI")):
    BENCHMARK_NAMES = [
        b for b in BENCHMARK_NAMES if b != "benchmark://cBench-v0/ghostscript"
    ]

env = gym.make("llvm-v0")
OBSERVATION_SPACE_NAMES = sorted(env.observation.spaces.keys())
REWARD_SPACE_NAMES = sorted(env.reward.spaces.keys())
env.close()


@pytest.fixture(scope="module")
def action_names() -> List[str]:
    """A list of every action."""
    return ACTION_NAMES


@pytest.fixture(scope="module", params=OBSERVATION_SPACE_NAMES)
def observation_space(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=REWARD_SPACE_NAMES)
def reward_space(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def benchmark_names() -> List[str]:
    """A list of every benchmarks."""
    return BENCHMARK_NAMES


@pytest.fixture(scope="module", params=ACTION_NAMES)
def action_name(request) -> str:
    """Enumerate the names of actions."""
    yield request.param


@pytest.fixture(scope="module", params=BENCHMARK_NAMES)
def benchmark_name(request) -> str:
    """Enumerate the names of benchmarks."""
    yield request.param


@pytest.fixture(scope="function")
def env() -> CompilerEnv:
    """Create an LLVM environment."""
    env = gym.make("llvm-v0")
    env.require_dataset("cBench-v0")
    try:
        yield env
    finally:
        env.close()
