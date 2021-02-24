# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/envs."""
import sys

import gym
import pytest

from compiler_gym.datasets import Dataset
from compiler_gym.envs import CompilerEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


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


def test_benchmark_constructor_arg(env: CompilerEnv):
    env.close()  # Fixture only required to pull in dataset.

    env = gym.make("llvm-v0", benchmark="cBench-v0/dijkstra")
    try:
        assert env.benchmark == "cBench-v0/dijkstra"
    finally:
        env.close()


if __name__ == "__main__":
    main()
