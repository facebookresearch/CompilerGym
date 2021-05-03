# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
from typing import Iterable, Optional

from compiler_gym.datasets import Dataset, TarDatasetWithManifest
from compiler_gym.envs.llvm.datasets.anghabench import AnghaBenchDataset
from compiler_gym.envs.llvm.datasets.cbench import CBenchDataset, CBenchLegacyDataset
from compiler_gym.envs.llvm.datasets.clgen import CLgenDataset
from compiler_gym.envs.llvm.datasets.csmith import CsmithBenchmark, CsmithDataset
from compiler_gym.envs.llvm.datasets.llvm_stress import LlvmStressDataset
from compiler_gym.envs.llvm.datasets.poj104 import POJ104Dataset, POJ104LegacyDataset
from compiler_gym.util.runfiles_path import site_data_path


class BlasDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://blas-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2"
            ],
            tar_sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0-manifest.bz2"
            ],
            manifest_sha256="6946437dcb0da5fad3ed8a7fd83eb4294964198391d5537b1310e22d7ceebff4",
            references={
                "Paper": "https://strum355.netsoc.co/books/PDF/Basic%20Linear%20Algebra%20Subprograms%20for%20Fortran%20Usage%20-%20BLAS%20(1979).pdf",
                "Homepage": "http://www.netlib.org/blas/",
            },
            license="BSD 3-Clause",
            strip_prefix="blas-v0",
            description="Basic linear algebra kernels",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class GitHubDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        manifest_url, manifest_sha256 = {
            "darwin": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0-macos-manifest.bz2",
                "10d933a7d608248be286d756b27813794789f7b87d8561c241d0897fb3238503",
            ),
            "linux": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0-linux-manifest.bz2",
                "aede9ca78657b4694ada9a4592d93f0bbeb3b3bd0fff3b537209850228480d3b",
            ),
        }[sys.platform]
        super().__init__(
            name="benchmark://github-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2"
            ],
            tar_sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
            manifest_urls=[manifest_url],
            manifest_sha256=manifest_sha256,
            license="CC BY 4.0",
            references={
                "Paper": "https://arxiv.org/pdf/2012.01470.pdf",
            },
            strip_prefix="github-v0",
            description="Compile-only C/C++ objects from GitHub",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class LinuxDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        manifest_url, manifest_sha256 = {
            "darwin": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0-macos-manifest.bz2",
                "dfc87b94c7a43e899e76507398a5af22178aebaebcb5d7e24e82088aeecb0690",
            ),
            "linux": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0-linux-manifest.bz2",
                "32ceb8576f683798010816ac605ee496f386ddbbe64be9e0796015d247a73f92",
            ),
        }[sys.platform]
        super().__init__(
            name="benchmark://linux-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2"
            ],
            tar_sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
            manifest_urls=[manifest_url],
            manifest_sha256=manifest_sha256,
            references={"Homepage": "https://www.linux.org/"},
            license="GPL-2.0",
            strip_prefix="linux-v0",
            description="Compile-only object files from C Linux kernel",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class MibenchDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://mibench-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2"
            ],
            tar_sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0-manifest.bz2"
            ],
            manifest_sha256="8ed985d685b48f444a3312cd84ccc5debda4a839850e442a3cdc93910ba0dc5f",
            references={
                "Paper": "http://vhosts.eecs.umich.edu/mibench/Publications/MiBench.pdf"
            },
            license="BSD 3-Clause",
            strip_prefix="mibench-v0",
            description="C benchmarks",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class NPBDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://npb-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2"
            ],
            tar_sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0-manifest.bz2"
            ],
            manifest_sha256="89eccb7f1b0b9e1f82b9b900b9f686ff5b189a2a67a4f8969a15901cd315dba2",
            references={
                "Paper": "http://optout.csc.ncsu.edu/~mueller/codeopt/codeopt05/projects/www4.ncsu.edu/~pgauria/csc791a/papers/NAS-95-020.pdf"
            },
            license="NASA Open Source Agreement v1.3",
            strip_prefix="npb-v0",
            description="NASA Parallel Benchmarks",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class OpenCVDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://opencv-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2"
            ],
            tar_sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0-manifest.bz2"
            ],
            manifest_sha256="8de96f722fab18f3a2a74db74b4038c7947fe8b3da867c9260206fdf5338cd81",
            references={
                "Paper": "https://mipro-proceedings.com/sites/mipro-proceedings.com/files/upload/sp/sp_008.pdf",
                "Homepage": "https://opencv.org/",
            },
            license="Apache 2.0",
            strip_prefix="opencv-v0",
            description="Compile-only object files from C++ OpenCV library",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class TensorFlowDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://tensorflow-v0",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0.tar.bz2"
            ],
            tar_sha256="f77dd1988c772e8359e1303cc9aba0d73d5eb27e0c98415ac3348076ab94efd1",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0-manifest.bz2"
            ],
            manifest_sha256="cffc45cd10250d483cb093dec913c8a7da64026686284cccf404623bd1da6da8",
            references={
                "Paper": "https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf",
                "Homepage": "https://www.tensorflow.org/",
            },
            license="Apache 2.0",
            strip_prefix="tensorflow-v0",
            description="Compile-only object files from C++ TensorFlow library",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


def get_llvm_datasets(site_data_base: Optional[Path] = None) -> Iterable[Dataset]:
    """Instantiate the builtin LLVM datasets.

    :param site_data_base: The root of the site data path.

    :return: An iterable sequence of :class:`Dataset
        <compiler_gym.datasets.Dataset>` instances.
    """
    site_data_base = site_data_base or site_data_path("llvm-v0")

    yield AnghaBenchDataset(site_data_base=site_data_base, sort_order=0)
    # Add legacy version of Anghabench using an old manifest.
    anghabench_v0_manifest_url, anghabench_v0_manifest_sha256 = {
        "darwin": (
            "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v0-macos-manifest.bz2",
            "39464256405aacefdb7550a7f990c9c578264c132804eec3daac091fa3c21bd1",
        ),
        "linux": (
            "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v0-linux-manifest.bz2",
            "a038d25d39ee9472662a9704dfff19c9e3512ff6a70f1067af85c5cb3784b477",
        ),
    }[sys.platform]
    yield AnghaBenchDataset(
        name="benchmark://anghabench-v0",
        site_data_base=site_data_base,
        sort_order=0,
        manifest_url=anghabench_v0_manifest_url,
        manifest_sha256=anghabench_v0_manifest_sha256,
        deprecated="Please use anghabench-v1",
    )
    yield BlasDataset(site_data_base=site_data_base, sort_order=0)
    yield CLgenDataset(site_data_base=site_data_base, sort_order=0)
    yield CBenchDataset(site_data_base=site_data_base, sort_order=-1)
    # Add legacy version of cbench-v1 in which the 'b' was capitalized. This
    # is deprecated and will be removed no earlier than v0.1.10.
    yield CBenchDataset(
        site_data_base=site_data_base,
        name="benchmark://cBench-v1",
        deprecated=(
            "Please use 'benchmark://cbench-v1' (note the lowercase name). "
            "The dataset is the same, only the name has changed"
        ),
        manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v1-manifest.bz2",
        manifest_sha256="635b94eeb2784dfedb3b53fd8f84517c3b4b95d851ddb662d4c1058c72dc81e0",
        sort_order=100,
    )
    yield CBenchLegacyDataset(site_data_base=site_data_base)
    yield CsmithDataset(site_data_base=site_data_base, sort_order=0)
    yield GitHubDataset(site_data_base=site_data_base, sort_order=0)
    yield LinuxDataset(site_data_base=site_data_base, sort_order=0)
    yield LlvmStressDataset(site_data_base=site_data_base, sort_order=0)
    yield MibenchDataset(site_data_base=site_data_base, sort_order=0)
    yield NPBDataset(site_data_base=site_data_base, sort_order=0)
    yield OpenCVDataset(site_data_base=site_data_base, sort_order=0)
    yield POJ104Dataset(site_data_base=site_data_base, sort_order=0)
    yield POJ104LegacyDataset(site_data_base=site_data_base, sort_order=100)
    yield TensorFlowDataset(site_data_base=site_data_base, sort_order=0)


__all__ = [
    "AnghaBenchDataset",
    "BlasDataset",
    "CBenchDataset",
    "CBenchLegacyDataset",
    "CLgenDataset",
    "CsmithBenchmark",
    "CsmithDataset",
    "get_llvm_datasets",
    "GitHubDataset",
    "LinuxDataset",
    "LlvmStressDataset",
    "MibenchDataset",
    "NPBDataset",
    "OpenCVDataset",
    "POJ104Dataset",
    "POJ104LegacyDataset",
    "TensorFlowDataset",
]
