# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/envs."""
import sys

import gym
import pytest

from compiler_gym.datasets import Dataset
from compiler_gym.envs import CompilerEnv, CompilerEnvState
from tests.test_main import main


def test_state_to_csv_from_csv():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, reward=1.5, commandline="-a -b -c"
    )
    state_from_csv = CompilerEnvState.from_csv(original_state.to_csv())

    assert state_from_csv.benchmark == "foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward == 1.5
    assert state_from_csv.commandline == "-a -b -c"


def test_state_to_csv_from_csv_no_reward():
    original_state = CompilerEnvState(
        benchmark="foo", walltime=100, commandline="-a -b -c"
    )
    state_from_csv = CompilerEnvState.from_csv(original_state.to_csv())

    assert state_from_csv.benchmark == "foo"
    assert state_from_csv.walltime == 100
    assert state_from_csv.reward is None
    assert state_from_csv.commandline == "-a -b -c"


def test_state_from_csv_empty():
    with pytest.raises(ValueError) as ctx:
        CompilerEnvState.from_csv("")

    assert str(ctx.value) == "Failed to parse input: ``"


def test_state_from_csv_invalid_format():
    with pytest.raises(ValueError) as ctx:
        CompilerEnvState.from_csv("abcdef")

    assert str(ctx.value).startswith("Failed to parse input: `abcdef`: ")


@pytest.fixture(scope="function")
def env() -> CompilerEnv:
    env = gym.make("llvm-v0")
    try:
        yield env
    finally:
        env.close()


def make_dataset(**kwargs) -> Dataset:
    default_kwargs = {
        "name": "test-dataset-v0",
        "url": "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
        "license": "MIT",
        "description": "A test dataset",
        "compiler": "llvm-10.0.0",
        "file_count": 10,
        "size_bytes": 2,
        "sha256": "e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
    }
    default_kwargs.update(kwargs)
    return Dataset(**default_kwargs)


def test_register_dataset(env: CompilerEnv):
    dataset = make_dataset()
    assert env.register_dataset(dataset)
    assert dataset.name in env.available_datasets


def test_register_dataset_matching_platform(env: CompilerEnv):
    platform = {"darwin": "macos"}.get(sys.platform, sys.platform)
    dataset = make_dataset(platforms=[platform])
    assert env.register_dataset(dataset)
    assert dataset.name in env.available_datasets


def test_register_dataset_different_platform(env: CompilerEnv):
    dataset = make_dataset(platforms=["not-a-real-platform"])
    assert not env.register_dataset(dataset)
    assert dataset.name not in env.available_datasets


def test_double_register_dataset(env: CompilerEnv):
    dataset = make_dataset()
    assert env.register_dataset(dataset)
    with pytest.raises(ValueError) as ctx:
        env.register_dataset(dataset)
    assert str(ctx.value) == f"Dataset already registered with name: {dataset.name}"


if __name__ == "__main__":
    main()
