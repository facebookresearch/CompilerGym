# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for serializing LLVM datasets."""
import pickle

from compiler_gym.datasets import Dataset
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


def test_pickle_dataset(dataset: Dataset):
    """Test that datasets can be pickled."""
    assert pickle.loads(pickle.dumps(dataset)) == dataset


def test_pickle_benchmark(dataset: Dataset):
    """Test that benchmarks can be pickled."""
    benchmark = next(dataset.benchmarks())
    assert pickle.loads(pickle.dumps(benchmark))


if __name__ == "__main__":
    main()
