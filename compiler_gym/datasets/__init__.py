# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manage datasets of benchmarks."""
from compiler_gym.datasets.benchmark import (
    Benchmark,
    BenchmarkInitError,
    BenchmarkSource,
)
from compiler_gym.datasets.dataset import (
    Dataset,
    DatasetInitError,
    activate,
    deactivate,
    delete,
    require,
)
from compiler_gym.datasets.datasets import Datasets
from compiler_gym.datasets.files_dataset import FilesDataset
from compiler_gym.datasets.tar_dataset import TarDataset, TarDatasetWithManifest
from compiler_gym.datasets.uri import BenchmarkUri

__all__ = [
    "activate",
    "Benchmark",
    "BenchmarkInitError",
    "BenchmarkSource",
    "BenchmarkUri",
    "Dataset",
    "DatasetInitError",
    "Datasets",
    "deactivate",
    "delete",
    "FilesDataset",
    "require",
    "TarDataset",
    "TarDatasetWithManifest",
]
