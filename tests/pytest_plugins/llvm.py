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

from compiler_gym.datasets import Dataset
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.envs.llvm.datasets.cbench import VALIDATORS
from compiler_gym.third_party import llvm
from compiler_gym.util.runfiles_path import runfiles_path

BENCHMARKS_LIST = Path(runfiles_path("compiler_gym/third_party/cbench/benchmarks.txt"))


def _read_list_file(path: Path) -> Iterable[str]:
    with open(str(path)) as f:
        for action in f:
            if action.strip():
                yield action.strip()


BENCHMARK_NAMES = list(_read_list_file(BENCHMARKS_LIST))

# Skip ghostscript on CI as it is just too heavy.
if bool(os.environ.get("CI")):
    BENCHMARK_NAMES = [
        b for b in BENCHMARK_NAMES if b != "benchmark://cbench-v1/ghostscript"
    ]

with gym.make("llvm-v0") as env:
    ACTION_NAMES = list(env.action_space.names)
    OBSERVATION_SPACE_NAMES = sorted(env.observation.spaces.keys())
    REWARD_SPACE_NAMES = sorted(env.reward.spaces.keys())
    DATASET_NAMES = sorted(d.name for d in env.datasets)


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


VALIDATABLE_CBENCH_URIS = [b for b in BENCHMARK_NAMES if b in VALIDATORS]
NON_VALIDATABLE_CBENCH_URIS = [b for b in BENCHMARK_NAMES if b not in VALIDATORS]


@pytest.fixture(scope="module", params=VALIDATABLE_CBENCH_URIS)
def validatable_cbench_uri(request) -> str:
    """Enumerate the names of benchmarks whose semantics can be validated."""
    yield request.param


@pytest.fixture(scope="module", params=NON_VALIDATABLE_CBENCH_URIS)
def non_validatable_cbench_uri(request) -> str:
    """Enumerate the names of benchmarks whose semantics cannot be validated."""
    yield request.param


@pytest.fixture(scope="function")
def env() -> LlvmEnv:
    """Create an LLVM environment."""
    with gym.make("llvm-v0") as env_:
        yield env_


@pytest.fixture(scope="module")
def llvm_opt() -> Path:
    """Test fixture that yields the path of opt."""
    return llvm.opt_path()


@pytest.fixture(scope="module")
def llvm_diff() -> Path:
    """Test fixture that yields the path of llvm-diff."""
    return llvm.llvm_diff_path()


@pytest.fixture(scope="module")
def clang() -> Path:
    """Test fixture that yields the path of clang."""
    return llvm.clang_path()


@pytest.fixture(scope="module", params=DATASET_NAMES)
def dataset_name(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=DATASET_NAMES)
def dataset(request) -> Dataset:
    with gym.make("llvm-v0") as env:
        return env.datasets[request.param]
