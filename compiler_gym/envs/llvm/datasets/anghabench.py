# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from pathlib import Path
from typing import Optional

from compiler_gym.datasets import Benchmark, TarDatasetWithManifest
from compiler_gym.envs.llvm.benchmarks import ClangInvocation


class AnghaBenchDataset(TarDatasetWithManifest):
    """A dataset of C programs curated from GitHub source code."""

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://anghabench-v0",
            description="Compile-only C/C++ functions extracted from GitHub",
            long_description_url="https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf",
            license="Unknown. See: https://github.com/brenocfg/AnghaBench/issues/1",
            site_data_base=site_data_base,
            manifest_url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-anghabench-v0-manifest.gz",
            manifest_sha256="3596ad6c7e336cf7a188f82a6f64f1cf566bd04e10b6a56710f7278892886d03",
            tar_url="https://github.com/brenocfg/AnghaBench/archive/d8034ac8562b8c978376008f4b33df01b8887b19.tar.gz",
            tar_sha256="85d068e4ce44f2581e3355ee7a8f3ccb92568e9f5bd338bc3a918566f3aff42f",
            strip_prefix="AnghaBench-d8034ac8562b8c978376008f4b33df01b8887b19",
            tar_compression="gz",
            benchmark_file_suffix=".bc",
            sort_order=sort_order,
        )

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        """
        :raise LookupError: If :code:`uri` is provided but does not exist.
        """
        self.install()
        if uri is None:
            return self.get_benchmark_by_index(self.random.integers(self.n))

        # THe absolute path of the file, without an extension.
        path_stem = self.dataset_root / uri[len(self.name) + 1 :]

        bitcode_abspath = Path(f"{path_stem}.bc")

        # If the file does not exist, compile it on-demand.
        if not bitcode_abspath.is_file():
            c_file_abspath = Path(f"{path_stem}.c")
            if not c_file_abspath.is_file():
                raise LookupError(
                    f"Benchmark not found: {uri} (file not found: {c_file_abspath})"
                )
            compile_cmd = ClangInvocation.from_c_file(c_file_abspath).command(
                outpath=bitcode_abspath
            )
            subprocess.check_call(compile_cmd)
            if not bitcode_abspath.is_file():
                raise OSError(
                    f"Compilation job did not produce expected output file '{bitcode_abspath}': {compile_cmd}"
                )

        return Benchmark.from_file(uri, bitcode_abspath)

    def compile_all(self):
        # TODO: parallelize
        for benchmark in self.benchmarks():
            pass
