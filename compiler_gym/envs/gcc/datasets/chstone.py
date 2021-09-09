# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, TarDatasetWithManifest
from compiler_gym.envs.gcc.gcc import Gcc
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.filesystem import atomic_file_write

URIS = [
    "benchmark://chstone-v0/adpcm",
    "benchmark://chstone-v0/aes",
    "benchmark://chstone-v0/blowfish",
    "benchmark://chstone-v0/dfadd",
    "benchmark://chstone-v0/dfdiv",
    "benchmark://chstone-v0/dfmul",
    "benchmark://chstone-v0/dfsin",
    "benchmark://chstone-v0/gsm",
    "benchmark://chstone-v0/jpeg",
    "benchmark://chstone-v0/mips",
    "benchmark://chstone-v0/motion",
    "benchmark://chstone-v0/sha",
]


# TODO(github.com/facebookresearch/CompilerGym/issues/325): This can be merged
# with the LLVM implementation.
class CHStoneDataset(TarDatasetWithManifest):
    """A dataset of C programs curated from GitHub source code.

    The dataset is from:

        Hara, Yuko, Hiroyuki Tomiyama, Shinya Honda, Hiroaki Takada, and Katsuya
        Ishii. "Chstone: A benchmark program suite for practical c-based
        high-level synthesis." In 2008 IEEE International Symposium on Circuits
        and Systems, pp. 1192-1195. IEEE, 2008.

    And is available at:

        http://www.ertl.jp/chstone/
    """

    def __init__(
        self,
        gcc_bin: Path,
        site_data_base: Path,
        sort_order: int = 0,
    ):
        super().__init__(
            name="benchmark://chstone-v0",
            description="Benchmarks for C-based High-Level Synthesis",
            references={
                "Paper": "http://www.yxi.com/applications/iscas2008-300_1027.pdf",
                "Homepage": "http://www.ertl.jp/chstone/",
            },
            license="Mixture of open source and public domain licenses",
            site_data_base=site_data_base,
            tar_urls=[
                "https://github.com/ChrisCummins/patmos_HLS/archive/e62d878ceb91e5a18007ca2e0a9602ee44ff7d59.tar.gz"
            ],
            tar_sha256="f7acab9d3c3dc7b971e62c8454bc909d84bddb6d0a96378e41beb94231739acb",
            strip_prefix="patmos_HLS-e62d878ceb91e5a18007ca2e0a9602ee44ff7d59/benchmarks/CHStone",
            tar_compression="gz",
            benchmark_file_suffix=".c",
            sort_order=sort_order,
            # We provide our own manifest.
            manifest_urls=[],
            manifest_sha256="",
        )
        self.gcc_bin = gcc_bin

    def benchmark_uris(self) -> Iterable[str]:
        yield from URIS

    @memoized_property
    def gcc(self):
        # Defer instantiation of Gcc from the constructor as it will fail if the
        # given Gcc is not available. Memoize the result as initialization is
        # expensive.
        return Gcc(bin=self.gcc_bin)

    def benchmark(self, uri: str) -> Benchmark:
        self.install()

        benchmark_name = uri[len(self.name) + 1 :]
        if not benchmark_name:
            raise LookupError(f"No benchmark specified: {uri}")

        # Most of the source files are named after the parent directory, but not
        # all.
        c_file_name = {
            "blowfish": "bf.c",
            "motion": "mpeg2.c",
            "sha": "sha_driver.c",
            "jpeg": "main.c",
        }.get(benchmark_name, f"{benchmark_name}.c")
        source_dir_path = self.dataset_root / benchmark_name
        source_path = source_dir_path / c_file_name
        preprocessed_path = source_dir_path / "src.c"

        # If the file does not exist, preprocess it on-demand.
        if not preprocessed_path.is_file():
            if not source_path.is_file():
                raise LookupError(
                    f"Benchmark not found: {uri} (file not found: {source_path})"
                )

            with atomic_file_write(preprocessed_path) as tmp_path:
                # TODO(github.com/facebookresearch/CompilerGym/issues/325): Send
                # over the unprocessed code to the service, have the service
                # preprocess. Until then, we do it client side with GCC having
                # to fixed by an environment variable
                self.gcc(
                    "-E",
                    "-o",
                    tmp_path.name,
                    c_file_name,
                    cwd=source_dir_path,
                    timeout=300,
                )

        return Benchmark.from_file(uri, preprocessed_path)

    @property
    def size(self) -> int:
        return len(URIS)
