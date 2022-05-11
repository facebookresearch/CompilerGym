# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Union

from compiler_gym.datasets import Dataset
from compiler_gym.envs.cgra.datasets.dfg_bench import GeneratedDFGs, GeneratedDFGs10, GeneratedDFGs15, GeneratedDFGs20, GeneratedDFGs5
from compiler_gym.util.runfiles_path import site_data_path


def _get_cgra_datasets(
    site_data_base: Optional[Path] = None
) -> Iterable[Dataset]:
    site_data_base = site_data_base or site_data_path("cgra-v0")

    yield GeneratedDFGs5(site_data_base=site_data_base)
    yield GeneratedDFGs10(site_data_base=site_data_base)
    yield GeneratedDFGs15(site_data_base=site_data_base)
    yield GeneratedDFGs20(site_data_base=site_data_base)


@lru_cache(maxsize=16)
def get_cgra_datasets(
    site_data_base: Optional[Path] = None
) -> List[Dataset]:
    """Instantiate the builtin cgra datasets.

    :param site_data_base: The root of the site data path.

    :return: An iterable sequence of :class:`Dataset
        <compiler_gym.datasets.Dataset>` instances.
    """
    return list(_get_cgra_datasets(site_data_base))


__all__ = [
    "GeneratedDFGs5",
    "GeneratedDFGs10",
    "GeneratedDFGs15",
    "GeneratedDFGs20",
    "get_cgra_datasets",
]
