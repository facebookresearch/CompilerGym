# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import List

from fasteners import InterProcessLock

from compiler_gym.datasets import Benchmark, BenchmarkInitError, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.util.download import download
from compiler_gym.util.filesystem import atomic_file_write
from compiler_gym.util.truncate import truncate


class CLgenDataset(TarDatasetWithManifest):
    """The CLgen dataset contains 1000 synthetically generated OpenCL kernels.

    The dataset is from:

        Cummins, Chris, Pavlos Petoumenos, Zheng Wang, and Hugh Leather.
        "Synthesizing benchmarks for predictive modeling." In 2017 IEEE/ACM
        International Symposium on Code Generation and Optimization (CGO),
        pp. 86-99. IEEE, 2017.

    And is available at:

        https://github.com/ChrisCummins/paper-synthesizing-benchmarks

    Installation
    ------------

    The CLgen dataset consists of OpenCL kernels that are compiled to LLVM-IR
    on-demand and cached. The first time each benchmark is used there is an
    overhead of compiling it from OpenCL to bitcode. This is a one-off cost.
    Compiling OpenCL to bitcode requires third party headers that are downloaded
    on the first call to :code:`install()`.
    """

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://clgen-v0",
            description="Synthetically generated OpenCL kernels",
            references={
                "Paper": "https://chriscummins.cc/pub/2017-cgo.pdf",
                "Homepage": "https://github.com/ChrisCummins/clgen",
            },
            license="GNU General Public License v3.0",
            site_data_base=site_data_base,
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-clgen-v0-manifest.bz2"
            ],
            manifest_sha256="d2bbc1da5a24a8cb03b604d1d8e59227b33bdfcd964ebe741ca8339f1c8d65cc",
            tar_urls=[
                "https://github.com/ChrisCummins/paper-synthesizing-benchmarks/raw/e45b6dffe9998f612624f05a6c4878ab4bcc84ec/data/clgen-1000.tar.bz2"
            ],
            tar_sha256="0bbd1b737f2537305e4db09b2971a5fa848b7c3a978bff6b570f45d1a488a72c",
            strip_prefix="clgen-1000/kernels",
            tar_compression="bz2",
            benchmark_file_suffix=".bc",
            sort_order=sort_order,
        )

        self._opencl_installed = False
        self._opencl_headers_installed_marker = (
            self._site_data_path / ".opencl-installed"
        )
        self.libclc_dir = self.site_data_path / "libclc"
        self.opencl_h_path = self.site_data_path / "opencl.h"

    def install(self):
        super().install()

        if not self._opencl_installed:
            self._opencl_installed = self._opencl_headers_installed_marker.is_file()

        if self._opencl_installed:
            return

        with self._tar_lock, InterProcessLock(self._tar_lockfile):
            # Repeat install check now that we are in the locked region.
            if self._opencl_headers_installed_marker.is_file():
                return

            # Download the libclc headers.
            shutil.rmtree(self.libclc_dir, ignore_errors=True)
            self.logger.info("Downloading OpenCL headers")
            tar_data = io.BytesIO(
                download(
                    "https://dl.fbaipublicfiles.com/compiler_gym/libclc-v0.tar.bz2",
                    sha256="f1c511f2ac12adf98dcc0fbfc4e09d0f755fa403c18f1fb1ffa5547e1fa1a499",
                )
            )
            with tarfile.open(fileobj=tar_data, mode="r:bz2") as arc:
                arc.extractall(str(self.site_data_path / "libclc"))

            # Download the OpenCL header.
            with open(self.opencl_h_path, "wb") as f:
                f.write(
                    download(
                        "https://github.com/ChrisCummins/clgen/raw/463c0adcd8abcf2432b24df0aca594b77a69e9d3/deeplearning/clgen/data/include/opencl.h",
                        sha256="f95b9f4c8b1d09114e491846d0d41425d24930ac167e024f45dab8071d19f3f7",
                    )
                )

            self._opencl_headers_installed_marker.touch()

    def benchmark(self, uri: str) -> Benchmark:
        self.install()

        benchmark_name = uri[len(self.name) + 1 :]
        if not benchmark_name:
            raise LookupError(f"No benchmark specified: {uri}")

        # The absolute path of the file, without an extension.
        path_stem = self.dataset_root / uri[len(self.name) + 1 :]

        bc_path, cl_path = Path(f"{path_stem}.bc"), Path(f"{path_stem}.cl")

        # If the file does not exist, compile it on-demand.
        if not bc_path.is_file():
            if not cl_path.is_file():
                raise LookupError(
                    f"Benchmark not found: {uri} (file not found: {cl_path}, path_stem {path_stem})"
                )

            # Compile the OpenCL kernel into a bitcode file.
            with atomic_file_write(bc_path) as tmp_bc_path:
                compile_command: List[str] = ClangInvocation.from_c_file(
                    cl_path,
                    copt=[
                        "-isystem",
                        str(self.libclc_dir),
                        "-include",
                        str(self.opencl_h_path),
                        "-target",
                        "nvptx64-nvidia-nvcl",
                        "-ferror-limit=1",  # Stop on first error.
                        "-w",  # No warnings.
                    ],
                ).command(outpath=tmp_bc_path)
                self.logger.debug("Exec %s", compile_command)
                clang = subprocess.Popen(
                    compile_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _, stderr = clang.communicate(timeout=300)

            if clang.returncode:
                compile_command = " ".join(compile_command)
                error = truncate(
                    stderr.decode("utf-8"), max_lines=20, max_line_len=20000
                )
                raise BenchmarkInitError(
                    f"Compilation job failed!\n"
                    f"Command: {compile_command}\n"
                    f"Error: {error}"
                )

        return BenchmarkWithSource.create(uri, bc_path, "kernel.cl", cl_path)
