# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Iterable, Optional

from compiler_gym.datasets import Dataset
from compiler_gym.envs.mlir.datasets.matmul import MatmulBenchmark, MatmulDataset
from compiler_gym.util.runfiles_path import site_data_path


def get_mlir_datasets(site_data_base: Optional[Path] = None) -> Iterable[Dataset]:
    """Instantiate the builtin LLVM datasets.
    :param site_data_base: The root of the site data path.
    :return: An iterable sequence of :class:`Dataset
        <compiler_gym.datasets.Dataset>` instances.
    """
    site_data_base = site_data_base or site_data_path("mlir-v0")

    yield MatmulDataset(site_data_base=site_data_base)


__all__ = ["MatmulDataset", "MatmulBenchmark"]
