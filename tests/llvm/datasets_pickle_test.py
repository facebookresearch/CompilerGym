# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for serializing LLVM datasets."""
import pickle

from compiler_gym.datasets import Dataset
from compiler_gym.envs.llvm import LlvmEnv
from tests.pytest_plugins.common import ci_only, skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]

# Installing all datasets on CI is expensive. Skip these tests, we define
# smaller versions of them below.


@skip_on_ci
def test_pickle_dataset(dataset: Dataset):
    """Test that datasets can be pickled."""
    assert pickle.loads(pickle.dumps(dataset)) == dataset


@skip_on_ci
def test_pickle_benchmark(dataset: Dataset):
    """Test that benchmarks can be pickled."""
    benchmark = next(dataset.benchmarks())
    assert pickle.loads(pickle.dumps(benchmark))


# Smaller versions of the above tests for CI.


@ci_only
def test_pickle_cbench_dataset(env: LlvmEnv):
    """Test that datasets can be pickled."""
    dataset = env.datasets["benchmark://cbench-v1"]
    assert pickle.loads(pickle.dumps(dataset)) == dataset


@ci_only
def test_pickle_cbench_benchmark(env: LlvmEnv):
    """Test that benchmarks can be pickled."""
    dataset = env.datasets["benchmark://cbench-v1"]
    benchmark = next(dataset.benchmarks())
    assert pickle.loads(pickle.dumps(benchmark))


if __name__ == "__main__":
    main()
