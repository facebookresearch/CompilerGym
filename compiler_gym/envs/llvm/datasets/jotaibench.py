# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import sys
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

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

    The AnghaBench dataset consists of C functions that are compiled to LLVM-IR
    on-demand and cached. The first time each benchmark is used there is an
    overhead of compiling it from C to bitcode. This is a one-off cost.
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
            "linux": (
                "http://cuda.dcc.ufmg.br/Jotai/src/Jotai_printRetVal.tar.bz2",
                "7d2c6326036d87a02318e81a29560f9bb4ead3dc33ffbd43e4fb2e95e09dd621",
            ),
        }[sys.platform]
        super().__init__(
            name=name or "benchmark://jotai-v1",
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
                "http://cuda.dcc.ufmg.br/Jotai/src/Jotai_printRetVal.tar.bz2"
            ],
            tar_sha256="7d2c6326036d87a02318e81a29560f9bb4ead3dc33ffbd43e4fb2e95e09dd621",
            strip_prefix="programs_no-ub_printableRetVal",
            tar_compression="bz2",
            benchmark_file_suffix=".bc",
            sort_order=sort_order,
            deprecated=deprecated,
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
    """TODO."""

    def __init__(
        self,
        site_data_base: Path,
    ):
        super().__init__(
            name="benchmark://jotai-runnable-v0",
            description="Runnable C/C++ functions extracted from GitHub",
            references={
                # TODO: Update these as necessary:
                "Paper": "https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
                "Homepage": "http://cuda.dcc.ufmg.br/angha/",
            },
            license="",  # TODO: License name.
            site_data_base=site_data_base,
            tar_urls=[
                # TODO: URL of where to download a tarball that contains the
                # benchmarks. For debugging, you could use something like
                # Dropbox or similar. For eventual production we can host them
                # in our S3 bucket for you.
                "http://cuda.dcc.ufmg.br/Jotai/src/Jotai_printRetVal.tar.bz2"
            ],
            tar_sha256="7d2c6326036d87a02318e81a29560f9bb4ead3dc33ffbd43e4fb2e95e09dd621",  # TODO: sha256sum of the above tarfile.
            strip_prefix="programs_no-ub_printableRetVal",  # TODO: If there is a subdirectory to strip, specify it here.
            tar_compression="bz2",
            # TODO: The file extension that is used to automatically enumerate
            # the benchmarks.
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

        # TODO: Here is where we specify how to build and run the benchmark.
        # This is what makes a benchmark "runnable".
        benchmark.proto.dynamic_config.MergeFrom(
            BenchmarkDynamicConfig(
                build_cmd=Command(
                    # TODO: Here is where you specify the command to build the
                    # benchmark. Assuming no deps, this should be fine.
                    argument=["$CC", "$IN"] + get_system_library_flags(),
                    timeout_seconds=60,
                    outfile=["a.out"],
                ),
                run_cmd=Command(
                    # TODO: Here is where you specify the command to build the
                    # benchmark. Assuming no deps, this should be fine.
                    argument=["./a.out 0"],
                    timeout_seconds=60,
                    # TODO: If the benchmark needs any input files, specify it here.
                    infile=[],
                    # TODO: If the benchmark produces any output files, specify it
                    # here.
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
