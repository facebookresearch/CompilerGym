# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the available LLVM datasets."""
from typing import Callable, Dict, Optional

from compiler_gym.datasets.dataset import Dataset

LLVM_DATASETS = [
    Dataset(
        name="blas-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=300,
        size_bytes=3969036,
        sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
    ),
    Dataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=7150112,
        sha256="498425efe8e335eb72623df50427eca87333d011f017ec3be5608883b4b7687a",
    ),
    Dataset(
        name="github-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2",
        license="CC BY 4.0",
        description="https://zenodo.org/record/4122437",
        compiler="llvm-10.0.0",
        file_count=50708,
        size_bytes=725974100,
        sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
    ),
    Dataset(
        name="linux-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2",
        license="GPL-2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=13920,
        size_bytes=516031044,
        sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
    ),
    Dataset(
        name="mibench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=40,
        size_bytes=238480,
        sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
    ),
    Dataset(
        name="npb-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2",
        license="NASA Open Source Agreement v1.3",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=122,
        size_bytes=2287444,
        sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
    ),
    Dataset(
        name="opencv-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=442,
        size_bytes=21903008,
        sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
    ),
    Dataset(
        name="poj104-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://sites.google.com/site/treebasedcnn/",
        compiler="llvm-10.0.0",
        file_count=49628,
        size_bytes=304207752,
        sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
    ),
    Dataset(
        name="polybench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-polybench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=27,
        size_bytes=162624,
        sha256="968087e68470e5b44dc687dae195143000c7478a23d6631b27055bb3bb3116b1",
    ),
    Dataset(
        name="tensorflow-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=1985,
        size_bytes=299697312,
        sha256="f77dd1988c772e8359e1303cc9aba0d73d5eb27e0c98415ac3348076ab94efd1",
    ),
]

# A map from benchmark name to a callback which takes as input an LlvmEnv
# instance and returns None if the environment is valid, else a string error
# message.
#
# TODO(cummins): Populate this map for cBench using the CK meta properties.
# See: https://github.com/ctuning/ai/blob/main/program/cbench-bzip2/.cm/meta.json
LLVM_BENCHMARK_VALIDATION_CALLBACKS: Dict[
    str, Callable[["LlvmEnv"], Optional[str]]
] = {}
