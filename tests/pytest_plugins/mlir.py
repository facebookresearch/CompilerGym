# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest fixtures for the MLIR CompilerGym environments."""
from pathlib import Path
from typing import Iterable

import gym
import pytest

from compiler_gym.datasets import Dataset
from compiler_gym.envs.mlir import MlirEnv


def _read_list_file(path: Path) -> Iterable[str]:
    with open(str(path)) as f:
        for action in f:
            if action.strip():
                yield action.strip()


with gym.make("mlir-v0") as env:
    OBSERVATION_SPACE_NAMES = sorted(env.observation.spaces.keys())
    REWARD_SPACE_NAMES = sorted(env.reward.spaces.keys())
    DATASET_NAMES = sorted(d.name for d in env.datasets)


@pytest.fixture(scope="module", params=OBSERVATION_SPACE_NAMES)
def observation_space(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=REWARD_SPACE_NAMES)
def reward_space(request) -> str:
    return request.param


@pytest.fixture(scope="function")
def env() -> MlirEnv:
    """Create an LLVM environment."""
    with gym.make("mlir-v0") as env_:
        yield env_


@pytest.fixture(scope="module", params=DATASET_NAMES)
def dataset_name(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=DATASET_NAMES)
def dataset(request) -> Dataset:
    with gym.make("mlir-v0") as env:
        return env.datasets[request.param]
