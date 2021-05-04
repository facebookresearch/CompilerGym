# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import logging
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from threading import Lock
from typing import Iterable, List

import numpy as np
from fasteners import InterProcessLock

from compiler_gym.datasets import Benchmark, BenchmarkSource, Dataset
from compiler_gym.datasets.benchmark import BenchmarkInitError, BenchmarkWithSource
from compiler_gym.datasets.dataset import DatasetInitError
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.truncate import truncate

# The maximum value for the --seed argument to csmith.
UINT_MAX = (2 ** 32) - 1


class CsmithBenchmark(BenchmarkWithSource):
    """A CSmith benchmark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._src = None

    @classmethod
    def create(cls, uri: str, bitcode: bytes, src: bytes) -> Benchmark:
        """Create a benchmark from paths."""
        benchmark = cls.from_file_contents(uri, bitcode)
        benchmark._src = src  # pylint: disable=protected-access
        return benchmark

    @memoized_property
    def sources(self) -> Iterable[BenchmarkSource]:
        return [
            BenchmarkSource(filename="source.c", contents=self._src),
        ]

    @property
    def source(self) -> str:
        """Return the single source file contents as a string."""
        return self._src.decode("utf-8")


class CsmithBuildError(DatasetInitError):
    """Error raised if :meth:`CsmithDataset.install()
    <compiler_gym.datasets.CsmithDataset.install>` fails."""

    def __init__(self, failing_stage: str, stdout: str, stderr: str):
        install_instructions = {
            "linux": "sudo apt install g++ m4",
            "darwin": "brew install m4",
        }[sys.platform]

        super().__init__(
            "\n".join(
                [
                    f"Failed to build Csmith from source, `{failing_stage}` failed.",
                    "You may be missing installation dependencies. Install them using:",
                    f"    {install_instructions}",
                    "See https://github.com/csmith-project/csmith#install-csmith for more details",
                    f"--- Start `{failing_stage}` logs: ---\n",
                    stdout,
                    stderr,
                ]
            )
        )


class CsmithDataset(Dataset):
    """A dataset which uses Csmith to generate programs.

    Csmith is a tool that can generate random conformant C99 programs. It is
    described in the publication:

        Yang, Xuejun, Yang Chen, Eric Eide, and John Regehr. "Finding and
        understanding bugs in C compilers." In Proceedings of the 32nd ACM
        SIGPLAN conference on Programming Language Design and Implementation
        (PLDI), pp. 283-294. 2011.

    For up-to-date information about Csmith, see:

        https://embed.cs.utah.edu/csmith/

    Note that Csmith is a tool that is used to find errors in compilers. As
    such, there is a higher likelihood that the benchmark cannot be used for an
    environment and that :meth:`env.reset()
    <compiler_gym.envs.CompilerEnv.reset>` will raise :class:`BenchmarkInitError
    <compiler_gym.datasets.BenchmarkInitError>`.

    Installation
    ------------

    Using the CsmithDataset requires building the Csmith binary from source.
    This is done automatically on the first call to :code:`install()`. Building
    Csmith requires a working C++ toolchain. Install the required dependencies
    using: :code:`sudo apt install -y g++ m4` on Linux, or :code:`brew install
    m4` on macOS. :class:`DatasetInitError
    <compiler_gym.datasets.DatasetInitError>` is raised if compilation fails.
    See the `Csmith repo
    <https://github.com/csmith-project/csmith#install-csmith>`_ for further
    details.
    """

    def __init__(self, site_data_base: Path, sort_order: int = 0):
        super().__init__(
            name="generator://csmith-v0",
            description="Random conformant C99 programs",
            references={
                "Paper": "http://web.cse.ohio-state.edu/~rountev.1/5343/pdf/pldi11.pdf",
                "Homepage": "https://embed.cs.utah.edu/csmith/",
            },
            license="BSD",
            site_data_base=site_data_base,
            sort_order=sort_order,
            benchmark_class=CsmithBenchmark,
        )
        self.csmith_path = self.site_data_path / "bin" / "csmith"
        csmith_include_dir = self.site_data_path / "include" / "csmith-2.3.0"

        self._installed = False
        self._build_lock = Lock()
        self._build_lockfile = self.site_data_path / ".build.LOCK"
        self._build_markerfile = self.site_data_path / ".built"

        # The command that is used to compile an LLVM-IR bitcode file from a
        # Csmith input. Reads from stdin, writes to stdout.
        self.clang_compile_command: List[str] = ClangInvocation.from_c_file(
            "-",  # Read from stdin.
            copt=[
                "-xc",
                "-ferror-limit=1",  # Stop on first error.
                "-w",  # No warnings.
                f"-I{csmith_include_dir}",  # Include the Csmith headers.
            ],
        ).command(
            outpath="-"
        )  # Write to stdout.

    @property
    def installed(self) -> bool:
        # Fast path for repeated checks to 'installed' without a disk op.
        if not self._installed:
            self._installed = self._build_markerfile.is_file()
        return self._installed

    def install(self) -> None:
        """Download and build the Csmith binary."""
        super().install()

        if self.installed:
            return

        with self._build_lock, InterProcessLock(self._build_lockfile):
            # Repeat the check to see if we have already installed the dataset
            # now that we have acquired the lock.
            if not self.installed:
                self.logger.info("Downloading and building Csmith")
                self._build_csmith(self.site_data_path, self.logger)
                self._build_markerfile.touch()

    @staticmethod
    def _build_csmith(install_root: Path, logger: logging.Logger):
        """Download, build, and install Csmith to the given directory."""
        tar_data = io.BytesIO(
            download(
                urls=[
                    "https://github.com/csmith-project/csmith/archive/refs/tags/csmith-2.3.0.tar.gz",
                ],
                sha256="ba871c1e5a05a71ecd1af514fedba30561b16ee80b8dd5ba8f884eaded47009f",
            )
        )
        # Csmith uses a standard `configure` + `make install` build process.
        with tempfile.TemporaryDirectory(
            dir=transient_cache_path("."), prefix="csmith-"
        ) as d:
            with tarfile.open(fileobj=tar_data, mode="r:gz") as arc:
                arc.extractall(d)

            # The path of the extracted sources.
            src_dir = Path(d) / "csmith-csmith-2.3.0"

            logger.debug("Configuring Csmith at %s", d)
            configure = subprocess.Popen(
                ["./configure", f"--prefix={install_root}"],
                cwd=src_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = configure.communicate(timeout=600)
            if configure.returncode:
                raise CsmithBuildError("./configure", stdout, stderr)

            logger.debug("Installing Csmith to %s", install_root)
            make = subprocess.Popen(
                ["make", "-j", "install"],
                cwd=src_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            stdout, stderr = make.communicate(timeout=600)
            if make.returncode:
                raise CsmithBuildError("make install", stdout, stderr)

    @property
    def size(self) -> int:
        # Actually 2^32 - 1, but practically infinite for all intents and
        # purposes.
        return float("inf")

    def benchmark_uris(self) -> Iterable[str]:
        return (f"{self.name}/{i}" for i in range(UINT_MAX))

    def benchmark(self, uri: str) -> CsmithBenchmark:
        return self.benchmark_from_seed(int(uri.split("/")[-1]))

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        seed = random_state.integers(UINT_MAX)
        return self.benchmark_from_seed(seed)

    def benchmark_from_seed(self, seed: int) -> CsmithBenchmark:
        """Get a benchmark from a uint32 seed.

        :param seed: A number in the range 0 <= n < 2^32.

        :return: A benchmark instance.
        """
        self.install()

        # Run csmith with the given seed and pipe the output to clang to
        # assemble a bitcode.
        self.logger.debug("Exec csmith --seed %d", seed)
        csmith = subprocess.Popen(
            [str(self.csmith_path), "--seed", str(seed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # Generate the C source.
        src, stderr = csmith.communicate(timeout=300)
        if csmith.returncode:
            error = truncate(stderr.decode("utf-8"), max_lines=20, max_line_len=100)
            raise OSError(f"Csmith failed with seed {seed}\nError: {error}")

        # Compile to IR.
        clang = subprocess.Popen(
            self.clang_compile_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = clang.communicate(src, timeout=300)

        if csmith.returncode:
            raise OSError(f"Csmith failed with seed {seed}")
        if clang.returncode:
            compile_cmd = " ".join(self.clang_compile_command)
            error = truncate(stderr.decode("utf-8"), max_lines=20, max_line_len=100)
            raise BenchmarkInitError(
                f"Compilation job failed!\n"
                f"Csmith seed: {seed}\n"
                f"Command: {compile_cmd}\n"
                f"Error: {error}"
            )

        return self.benchmark_class.create(f"{self.name}/{seed}", stdout, src)
