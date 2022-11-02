# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from concurrent.futures import as_completed
from pathlib import Path

from compiler_gym.datasets import Benchmark, TarDataset, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.llvm.llvm_benchmark import (
    ClangInvocation,
    get_system_library_flags,
)
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command
from compiler_gym.util import thread_pool
from compiler_gym.util.filesystem import atomic_file_write


class JotaiBenchDataset(TarDatasetWithManifest):
    """A dataset of C programs curated from GitHub source code.

    The dataset is from:

        da Silva, Anderson Faustino, Bruno Conde Kind, José Wesley de Souza
        Magalhaes, Jerônimo Nunes Rocha, Breno Campos Ferreira Guimaraes, and
        Fernando Magno Quinão Pereira. "ANGHABENCH: A Suite with One Million
        Compilable C Benchmarks for Code-Size Reduction." In 2021 IEEE/ACM
        International Symposium on Code Generation and Optimization (CGO),
        pp. 378-390. IEEE, 2021.

    And is available at:

        http://cuda.dcc.ufmg.br/Jotai/src/

    Installation
    ------------

    The JotaiBench dataset consists of C functions that are compiled to LLVM-IR
    on-demand and cached. The first time each benchmark is used there is an
    overhead of compiling it from C to bitcode. This is a one-off cost.
    """

    def __init__(
        self,
        site_data_base: Path,
    ):
        super().__init__(
            name="benchmark://jotaibench-v0",
            description="Compile-only C/C++ functions extracted from GitHub",
            references={
                "Paper": "https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
                "Homepage": "http://cuda.dcc.ufmg.br/angha/",
            },
            license="GNU General Public License v3.0 (GPLv3)",
            site_data_base=site_data_base,
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-jotaibench-v0.bz2"
            ],
            manifest_sha256="ac4ee456e52073964d472d3e3969058b2f3052f8a4b402719013a3c603eb4b62",
            tar_urls=[
                "https://github.com/ChrisCummins/jotai-benchmarks/raw/ca26ccd27afecf38919c1e101c64e3cc17e39631/benchmarks/jotaibench.bz2"
            ],
            tar_sha256="b5a51af3d4e2f77a66001635ec64ed321e0ece19873c4a888040859af7556401",
            strip_prefix="jotaibench/jotaibench-v0",
            tar_compression="bz2",
            benchmark_file_suffix=".c",
            sort_order=0,
        )

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        self.install()

        benchmark_name = uri.path[1:]
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


class JotaiBenchRunnableDataset(TarDataset):
    def __init__(
        self,
        site_data_base: Path,
    ):
        super().__init__(
            name="benchmark://jotai-runnable-v0",
            description="Runnable C/C++ functions extracted from GitHub",
            references={
                "Paper": "https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
                "Homepage": "http://cuda.dcc.ufmg.br/angha/",
            },
            license="GNU General Public License v3.0 (GPLv3)",
            site_data_base=site_data_base,
            tar_urls=[
                "https://github.com/lac-dcc/jotai-benchmarks/blob/main/benchmarks/jotaibench.bz2?raw=true"
            ],
            tar_sha256="b5a51af3d4e2f77a66001635ec64ed321e0ece19873c4a888040859af7556401",
            strip_prefix="jotaibench-v0",
            tar_compression="bz2",
            benchmark_file_suffix=".c",
        )

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        self.install()

        benchmark_name = uri.path[1:]
        if not benchmark_name:
            raise LookupError(f"No benchmark specified: {uri}")

        # The absolute path of the file, without an extension.
        path_stem = self.dataset_root / benchmark_name

        bitcode_abspath = Path(f"{path_stem}.bc")
        c_file_abspath = Path(f"{path_stem}.c")

        # If the file does not exist, compile it to a bitcode file on-demand.
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

        benchmark = BenchmarkWithSource.create(
            uri, bitcode_abspath, "function.c", c_file_abspath
        )

        # This is what makes a benchmark "runnable".
        benchmark.proto.dynamic_config.MergeFrom(
            BenchmarkDynamicConfig(
                build_cmd=Command(
                    argument=["$CC", "$IN"] + get_system_library_flags(),
                    timeout_seconds=30,
                    outfile=["a.out"],
                ),
                run_cmd=Command(
                    argument=["./a.out 0"],
                    timeout_seconds=30,
                    infile=[],
                    outfile=[],
                ),
            )
        )

        return benchmark

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
