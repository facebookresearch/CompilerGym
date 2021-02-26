# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Iterable, Optional

from compiler_gym.datasets import Dataset, TarDatasetWithManifest
from compiler_gym.envs.llvm.datasets.anghabench import AnghaBenchDataset
from compiler_gym.envs.llvm.datasets.cbench import CBenchDataset, CBenchLegacyDataset
from compiler_gym.envs.llvm.datasets.llvm_stress import LlvmStressDataset
from compiler_gym.util.runfiles_path import site_data_path


class BlasDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://blas-v0",
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
            tar_sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0-manifest.gz",
            manifest_sha256="1d561808bc80e72a33f13b376c10502f1af2645ed6f0fb1851de1b746402db01",
            long_description_url="https://github.com/spcl/ncc/tree/master/data",
            license="BSD 3-Clause",
            strip_prefix="blas-v0",
            description="Basic linear algebra kernels",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class GitHubDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://github-v0",
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2",
            tar_sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0-manifest.gz",
            manifest_sha256="6d0ed47f8c70868db62ae9e3d2f33dbad9fda5ef1cfe99a9855eef4618ddef1b",
            license="CC BY 4.0",
            long_description_url="https://github.com/ctuning/ctuning-programs",
            strip_prefix="github-v0",
            description="Compile-only C/C++ objects from GitHub",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class LinuxDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://linux-v0",
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2",
            tar_sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0-manifest.gz",
            manifest_sha256="6b45716ca142950e42958634366626d06f02e73d37ddce225b3ef55468011aa8",
            long_description_url="https://github.com/spcl/ncc/tree/master/data",
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
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2",
            tar_sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0-manifest.gz",
            manifest_sha256="059bc81b92d5942ac0ea74664b8268d1e64f5489dca66992b50b5ef7b527264e",
            long_description_url="https://github.com/ctuning/ctuning-programs",
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
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2",
            tar_sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0-manifest.gz",
            manifest_sha256="edd5cf0863db49cee6551a7cabefc08e931295f3ba1e2990705f05442eb5ebbc",
            long_description_url="https://github.com/spcl/ncc/tree/master/data",
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
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2",
            tar_sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0-manifest.gz",
            manifest_sha256="e5fc1afbfbb978b2e6a5d4d7f3ffed7c612887fbdc1af5f5cba4d0ab29c3ed9b",
            long_description_url="https://github.com/spcl/ncc/tree/master/data",
            license="Apache 2.0",
            strip_prefix="opencv-v0",
            description="Compile-only object files from C++ OpenCV library",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class POJ104Dataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://poj104-v0",
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
            tar_sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0-manifest.gz",
            manifest_sha256="ca68aec704d054a26046bc82aff17938e49f9083078dacc5f042c6051f2d2711",
            long_description_url="https://sites.google.com/site/treebasedcnn/",
            license="BSD 3-Clause",
            strip_prefix="poj104-v0",
            description="Solutions to programming programs",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


class TensorflowDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://tensorflow-v0",
            tar_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0.tar.bz2",
            tar_sha256="f77dd1988c772e8359e1303cc9aba0d73d5eb27e0c98415ac3348076ab94efd1",
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0-manifest.gz",
            manifest_sha256="a78751b4562f27d330e4c20f34b5f1e670fcbe1a92172f0d01c6eba49b182576",
            long_description_url="https://github.com/spcl/ncc/tree/master/data",
            license="Apache 2.0",
            strip_prefix="tensorflow-v0",
            description="Compile-only object files from C++ TensorFlow library",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )


def get_llvm_datasets(site_data_base: Optional[Path] = None) -> Iterable[Dataset]:
    site_data_base = site_data_base or site_data_path("llvm/10.0.0/bitcode_benchmarks")

    yield AnghaBenchDataset(site_data_base=site_data_base, sort_order=0)
    yield BlasDataset(site_data_base=site_data_base, sort_order=0)
    yield CBenchDataset(site_data_base=site_data_base, sort_order=-1)
    # Add legacy version of cbench-v1 in which the 'b' was capitalized. This
    # is deprecated and will be removed no earlier than v0.1.10.
    yield CBenchDataset(
        site_data_base=site_data_base,
        name="benchmark://cBench-v1",
        hidden=True,
        manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v1-manifest.gz",
        manifest_sha256="455636dde21013fb593afd47c3dc2d25401a8a8cfff5dde01d2e416f039149ba",
    )
    yield CBenchLegacyDataset(site_data_base=site_data_base)
    yield GitHubDataset(site_data_base=site_data_base, sort_order=0)
    yield LinuxDataset(site_data_base=site_data_base, sort_order=0)
    yield LlvmStressDataset(site_data_base=site_data_base, sort_order=0)
    yield MibenchDataset(site_data_base=site_data_base, sort_order=0)
    yield NPBDataset(site_data_base=site_data_base, sort_order=0)
    yield OpenCVDataset(site_data_base=site_data_base, sort_order=0)
    yield POJ104Dataset(site_data_base=site_data_base, sort_order=0)
    yield TensorflowDataset(site_data_base=site_data_base, sort_order=0)


__all__ = [
    "AnghaBenchDataset",
    "BlasDataset",
    "CBenchDataset",
    "CBenchLegacyDataset",
    "get_llvm_datasets",
    "GitHubDataset",
    "LinuxDataset",
    "LlvmStressDataset",
    "MibenchDataset",
    "NPBDataset",
    "OpenCVDataset",
    "POJ104Dataset",
    "TensorflowDataset",
]
