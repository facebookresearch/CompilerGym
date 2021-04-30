# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import sys
from concurrent.futures import as_completed
from pathlib import Path

from compiler_gym.datasets import Benchmark, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.util import thread_pool
from compiler_gym.util.filesystem import atomic_file_write


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

    Installation
    ------------

    The AnghaBench dataset consists of C functions that are compiled to LLVM-IR
    on-demand and cached. The first time each benchmark is used there is an
    overhead of compiling it from C to bitcode. This is a one-off cost.
    """

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        manifest_url, manifest_sha256 = {
            "darwin": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v0-macos-manifest.bz2",
                "39464256405aacefdb7550a7f990c9c578264c132804eec3daac091fa3c21bd1",
            ),
            "linux": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v0-linux-manifest.bz2",
                "a038d25d39ee9472662a9704dfff19c9e3512ff6a70f1067af85c5cb3784b477",
            ),
        }[sys.platform]
        super().__init__(
            name="benchmark://anghabench-v0",
            description="Compile-only C/C++ functions extracted from GitHub",
            references={
                "Paper": "https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
                "Homepage": "http://cuda.dcc.ufmg.br/angha/",
            },
            license="Unknown. See: https://github.com/brenocfg/AnghaBench/issues/1",
            site_data_base=site_data_base,
            manifest_urls=[manifest_url],
            manifest_sha256=manifest_sha256,
            tar_urls=[
                "https://github.com/brenocfg/AnghaBench/archive/d8034ac8562b8c978376008f4b33df01b8887b19.tar.gz"
            ],
            tar_sha256="85d068e4ce44f2581e3355ee7a8f3ccb92568e9f5bd338bc3a918566f3aff42f",
            strip_prefix="AnghaBench-d8034ac8562b8c978376008f4b33df01b8887b19",
            tar_compression="gz",
            benchmark_file_suffix=".bc",
            sort_order=sort_order,
        )

    def benchmark(self, uri: str) -> Benchmark:
        self.install()

        benchmark_name = uri[len(self.name) + 1 :]
        if not benchmark_name:
            raise LookupError(f"No benchmark specified: {uri}")

        # The absolute path of the file, without an extension.
        path_stem = self.dataset_root / benchmark_name

        bitcode_abspath = Path(f"{path_stem}.bc")
        c_file_abspath = Path(f"{path_stem}.c")

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
