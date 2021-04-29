# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import sys
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

from compiler_gym.datasets import Benchmark, BenchmarkInitError, TarDatasetWithManifest
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.util import thread_pool
from compiler_gym.util.download import download
from compiler_gym.util.filesystem import atomic_file_write
from compiler_gym.util.truncate import truncate


class POJ104Dataset(TarDatasetWithManifest):
    """The POJ-104 dataset contains 52000 C++ programs implementing 104
    different algorithms with 500 examples of each.

    The dataset is from:

        Lili Mou, Ge Li, Lu Zhang, Tao Wang, Zhi Jin. "Convolutional neural
        networks over tree structures for programming language processing." To
        appear in Proceedings of 30th AAAI Conference on Artificial
        Intelligence, 2016.

    And is available at:

        https://sites.google.com/site/treebasedcnn/
    """

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        manifest_url, manifest_sha256 = {
            "darwin": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v1-macos-manifest.bz2",
                "74db443f225478933dd0adf3f821fd4e615089eeaa90596c19d9d1af7006a801",
            ),
            "linux": (
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v1-linux-manifest.bz2",
                "ee6253ee826e171816105e76fa78c0d3cbd319ef66e10da4bcf9cf8a78e12ab9",
            ),
        }[sys.platform]
        super().__init__(
            name="benchmark://poj104-v1",
            tar_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v1.tar.gz",
                "https://drive.google.com/u/0/uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU&export=download",
            ],
            tar_sha256="c0b8ef3ee9c9159c882dc9337cb46da0e612a28e24852a83f8a1cd68c838f390",
            tar_compression="gz",
            manifest_urls=[manifest_url],
            manifest_sha256=manifest_sha256,
            references={
                "Paper": "https://ojs.aaai.org/index.php/AAAI/article/download/10139/9998",
                "Homepage": "https://sites.google.com/site/treebasedcnn/",
            },
            license="BSD 3-Clause",
            strip_prefix="ProgramData",
            description="Solutions to programming programs",
            benchmark_file_suffix=".txt",
            site_data_base=site_data_base,
            sort_order=sort_order,
        )

    def benchmark(self, uri: Optional[str] = None) -> Benchmark:
        self.install()
        if uri is None or len(uri) <= len(self.name) + 1:
            return self._get_benchmark_by_index(self.random.integers(self.size))

        # The absolute path of the file, without an extension.
        path_stem = self.dataset_root / uri[len(self.name) + 1 :]

        # If the file does not exist, compile it on-demand.
        bitcode_path = Path(f"{path_stem}.bc")
        cc_file_path = Path(f"{path_stem}.txt")

        if not bitcode_path.is_file():
            if not cc_file_path.is_file():
                raise LookupError(
                    f"Benchmark not found: {uri} (file not found: {cc_file_path})"
                )

            # Load the C++ source into memory and pre-process it.
            with open(cc_file_path) as f:
                src = self.preprocess_poj104_source(f.read())

            # Compile the C++ source into a bitcode file.
            with atomic_file_write(bitcode_path) as tmp_bitcode_path:
                compile_cmd = ClangInvocation.from_c_file(
                    "-",
                    copt=[
                        "-xc++",
                        "-ferror-limit=1",  # Stop on first error.
                        "-w",  # No warnings.
                        # Some of the programs use the gets() function that was
                        # deprecated in C++11 and removed in C++14.
                        "-std=c++11",
                    ],
                ).command(outpath=tmp_bitcode_path)
                self.logger.debug("Exec %s", compile_cmd)
                clang = subprocess.Popen(
                    compile_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _, stderr = clang.communicate(src.encode("utf-8"), timeout=300)

            if clang.returncode:
                compile_cmd = " ".join(compile_cmd)
                error = truncate(stderr.decode("utf-8"), max_lines=20, max_line_len=100)
                raise BenchmarkInitError(
                    f"Compilation job failed!\n"
                    f"Command: {compile_cmd}\n"
                    f"Error: {error}"
                )
            if not bitcode_path.is_file():
                raise BenchmarkInitError(
                    f"Compilation job failed to produce output file!\nCommand: {compile_cmd}"
                )

        return BenchmarkWithSource.create(uri, bitcode_path, "source.cc", cc_file_path)

    @staticmethod
    def preprocess_poj104_source(src: str) -> str:
        """Pre-process a POJ-104 C++ source file for compilation."""
        # Clean up declaration of main function. Many are missing a return type
        # declaration, or use an incorrect void return type.
        src = src.replace("void main", "int main")
        src = src.replace("\nmain", "int main")
        if src.startswith("main"):
            src = f"int {src}"

        # Pull in the standard library.
        if sys.platform == "linux":
            header = "#include <bits/stdc++.h>\n" "using namespace std;\n"
        else:
            # Download a bits/stdc++ implementation for macOS.
            header = download(
                "https://raw.githubusercontent.com/tekfyl/bits-stdc-.h-for-mac/e1193f4470514d82ea19c3cc1357116fadaa2a4e/stdc%2B%2B.h",
                sha256="b4d9b031d56d89a2b58b5ed80fa9943aa92420d6aed0835747c9a5584469afeb",
            ).decode("utf-8")

        # These defines provide values for commonly undefined symbols. Defining
        # these macros increases the number of POJ-104 programs that compile
        # from 49,302 to 49,821 (+519) on linux.
        defines = "#define LEN 128\n" "#define MAX_LENGTH 1024\n" "#define MAX 1024\n"

        return header + defines + src

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
                f"\r\033[KCompiled {i} of {n} programs ({i/n:.2%} complete)",
                flush=True,
                end="",
            )


class POJ104LegacyDataset(TarDatasetWithManifest):
    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="benchmark://poj104-v0",
            tar_urls="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
            tar_sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
            manifest_urls=[
                "https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0-manifest.bz2"
            ],
            manifest_sha256="ac3eaaad7d2878d871ed2b5c72a3f39c058ea6694989af5c86cd162414db750b",
            references={
                "Paper": "https://ojs.aaai.org/index.php/AAAI/article/download/10139/9998",
                "Homepage": "https://sites.google.com/site/treebasedcnn/",
            },
            license="BSD 3-Clause",
            strip_prefix="poj104-v0",
            description="Solutions to programming programs",
            benchmark_file_suffix=".bc",
            site_data_base=site_data_base,
            sort_order=sort_order,
            deprecated="Please update to benchmark://poj104-v1.",
        )
