# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
from typing import Optional

from compiler_gym.datasets import TarDatasetWithManifest


# TODO(github.com/facebookresearch/CompilerGym/issues/325): This can be merged
# with the LLVM implementation.
class AnghaBenchDataset(TarDatasetWithManifest):
    """A dataset of C programs curated from GitHub source code.

    The dataset is from:

        da Silva, Anderson Faustino, Bruno Conde Kind, José Wesley de Souza
        Magalhaes, Jerônimo Nunes Rocha, Breno Campos Ferreira Guimaraes, and
        Fernando Magno Quinão Pereira. "ANGHABENCH: A Suite with One Million
        Compilable C Benchmarks for Code-Size Reduction." In 2021 IEEE/ACM
        International Symposium on Code Generation and Optimization (CGO),
        pp. 378-390. IEEE, 2021.

    And is available at:

        http://cuda.dcc.ufmg.br/angha/home
    """

    def __init__(
        self,
        site_data_base: Path,
        sort_order: int = 0,
        manifest_url: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        deprecated: Optional[str] = None,
        name: Optional[str] = None,
    ):
        manifest_url_, manifest_sha256_ = {
            "darwin": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v1-macos-manifest.bz2",
                "96ead63da5f8efa07fd0370f0c6e452b59bed840828b8b19402102b1ce3ee109",
            ),
            "linux": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v1-linux-manifest.bz2",
                "14df85f650199498cf769715e9f0d7841d09f9fa62a95b8ecc242bdaf227f33a",
            ),
        }[sys.platform]
        super().__init__(
            name=name or "benchmark://anghabench-v1",
            description="Compile-only C/C++ functions extracted from GitHub",
            references={
                "Paper": "https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
                "Homepage": "http://cuda.dcc.ufmg.br/angha/",
            },
            license="Unknown. See: https://github.com/brenocfg/AnghaBench/issues/1",
            site_data_base=site_data_base,
            manifest_urls=[manifest_url or manifest_url_],
            manifest_sha256=manifest_sha256 or manifest_sha256_,
            tar_urls=[
                "https://github.com/brenocfg/AnghaBench/archive/d8034ac8562b8c978376008f4b33df01b8887b19.tar.gz"
            ],
            tar_sha256="85d068e4ce44f2581e3355ee7a8f3ccb92568e9f5bd338bc3a918566f3aff42f",
            strip_prefix="AnghaBench-d8034ac8562b8c978376008f4b33df01b8887b19",
            tar_compression="gz",
            benchmark_file_suffix=".c",
            sort_order=sort_order,
            deprecated=deprecated,
        )
