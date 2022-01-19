# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from concurrent.futures import as_completed
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.util import thread_pool
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
            benchmark_file_suffix=".bc",
            sort_order=sort_order,
            # We provide our own manifest.
            manifest_urls=[],
            manifest_sha256="",
        )

    def benchmark_uris(self) -> Iterable[str]:
        yield from URIS

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        self.install()

        benchmark_name = uri.path[1:]
        if not benchmark_name:
            raise LookupError(f"No benchmark specified: {uri}")

        bitcode_abspath = self.dataset_root / f"{benchmark_name}.bc"

        # Most of the source files are named after the parent directory, but not
        # all.
        c_file_name = {
            "blowfish": "bf.c",
            "motion": "mpeg2.c",
            "sha": "sha_driver.c",
            "jpeg": "main.c",
        }.get(benchmark_name, f"{benchmark_name}.c")
        c_file_abspath = self.dataset_root / benchmark_name / c_file_name

        # If the file does not exist, compile it on-demand.
        if not bitcode_abspath.is_file():
            if not c_file_abspath.is_file():
                raise LookupError(
                    f"Benchmark not found: {uri} (file not found: {c_file_abspath})"
                )

            with atomic_file_write(bitcode_abspath) as tmp_path:
                compile_cmd = ClangInvocation.from_c_file(
                    c_file_abspath,
                    copt=[
                        "-ferror-limit=1",  # Stop on first error.
                        "-w",  # No warnings.
                    ],
                ).command(outpath=tmp_path)
                subprocess.check_call(compile_cmd, timeout=300)

        return BenchmarkWithSource.create(
            uri, bitcode_abspath, "function.c", c_file_abspath
        )

    @property
    def size(self) -> int:
        return len(URIS)

    def compile_all(self):
        n = self.size
        executor = thread_pool.get_thread_pool_executor()
        # Since the dataset is lazily compiled, simply iterating over the full
        # set of URIs will compile everything. Do this in parallel.
        futures = (
            executor.submit(self.benchmark, uri) for uri in self.benchmark_uris()
        )
        for i, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(
                f"\r\033[KCompiled {i} of {n} programs ({i/n:.1%} complete)",
                flush=True,
                end="",
            )
