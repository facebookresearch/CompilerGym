# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional, Union

import numpy as np
from fasteners import InterProcessLock

from compiler_gym.datasets import Benchmark, BenchmarkSource, Dataset
from compiler_gym.datasets.benchmark import BenchmarkWithSource
from compiler_gym.envs.gcc.gcc import Gcc
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.shell_format import plural
from compiler_gym.util.truncate import truncate

logger = logging.getLogger(__name__)

# The maximum value for the --seed argument to csmith.
UINT_MAX = (2 ** 32) - 1

_CSMITH_BIN = runfiles_path("compiler_gym/third_party/csmith/csmith/bin/csmith")
_CSMITH_INCLUDES = runfiles_path(
    "compiler_gym/third_party/csmith/csmith/include/csmith-2.3.0"
)
_CSMITH_INSTALL_LOCK = Lock()


# TODO(github.com/facebookresearch/CompilerGym/issues/325): This can be merged
# with the LLVM implementation.
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
    """

    def __init__(
        self,
        gcc_bin: Union[Path, str],
        site_data_base: Path,
        sort_order: int = 0,
        csmith_bin: Optional[Path] = None,
        csmith_includes: Optional[Path] = None,
    ):
        """Constructor.

        :param site_data_base: The base path of a directory that will be used to
            store installed files.

        :param sort_order: An optional numeric value that should be used to
            order this dataset relative to others. Lowest value sorts first.

        :param csmith_bin: The path of the Csmith binary to use. If not
            provided, the version of Csmith shipped with CompilerGym is used.

        :param csmith_includes: The path of the Csmith includes directory. If
            not provided, the includes of the Csmith shipped with CompilerGym is
            used.
        """
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
        self.gcc_bin = gcc_bin
        self.csmith_bin_path = csmith_bin or _CSMITH_BIN
        self.csmith_includes_path = csmith_includes or _CSMITH_INCLUDES
        self._install_lockfile = self.site_data_path / ".install.LOCK"

    @property
    def size(self) -> int:
        # Actually 2^32 - 1, but practically infinite for all intents and
        # purposes.
        return 0

    @memoized_property
    def gcc(self):
        # Defer instantiation of Gcc from the constructor as it will fail if the
        # given Gcc is not available. Memoize the result as initialization is
        # expensive.
        return Gcc(bin=self.gcc_bin)

    def benchmark_uris(self) -> Iterable[str]:
        return (f"{self.name}/{i}" for i in range(UINT_MAX))

    def benchmark(self, uri: str) -> CsmithBenchmark:
        return self.benchmark_from_seed(int(uri.split("/")[-1]))

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        seed = random_state.integers(UINT_MAX)
        return self.benchmark_from_seed(seed)

    @property
    def installed(self) -> bool:
        return super().installed and (self.site_data_path / "includes").is_dir()

    def install(self) -> None:
        super().install()

        if self.installed:
            return

        with _CSMITH_INSTALL_LOCK, InterProcessLock(self._install_lockfile):
            if (self.site_data_path / "includes").is_dir():
                return

            # Copy the Csmith headers into the dataset's site directory path because
            # in bazel builds this includes directory is a symlink, and we need
            # actual files that we can use in a docker volume.
            shutil.copytree(
                self.csmith_includes_path,
                self.site_data_path / "includes.tmp",
            )
            # Atomic directory rename to prevent race on install().
            (self.site_data_path / "includes.tmp").rename(
                self.site_data_path / "includes"
            )

    def benchmark_from_seed(
        self, seed: int, max_retries: int = 3, retry_count: int = 0
    ) -> CsmithBenchmark:
        """Get a benchmark from a uint32 seed.

        :param seed: A number in the range 0 <= n < 2^32.

        :return: A benchmark instance.

        :raises OSError: If Csmith fails.

        :raises BenchmarkInitError: If the C program generated by Csmith cannot
            be lowered to LLVM-IR.
        """
        if retry_count >= max_retries:
            raise OSError(
                f"Csmith failed after {retry_count} {plural(retry_count, 'attempt', 'attempts')} "
                f"with seed {seed}"
            )

        self.install()

        # Run csmith with the given seed and pipe the output to clang to
        # assemble a bitcode.
        logger.debug("Exec csmith --seed %d", seed)
        csmith = subprocess.Popen(
            [str(self.csmith_bin_path), "--seed", str(seed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Generate the C source.
        src, stderr = csmith.communicate(timeout=300)
        if csmith.returncode:
            try:
                stderr = "\n".join(
                    truncate(stderr.decode("utf-8"), max_line_len=200, max_lines=20)
                )
                logger.warning("Csmith failed with seed %d: %s", seed, stderr)
            except UnicodeDecodeError:
                # Failed to interpret the stderr output, generate a generic
                # error message.
                logger.warning("Csmith failed with seed %d", seed)
            return self.benchmark_from_seed(
                seed, max_retries=max_retries, retry_count=retry_count + 1
            )

        # Pre-process the source.
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = f"{tmpdir}/src.c"
            with open(src_file, "wb") as f:
                f.write(src)

            preprocessed_src = self.gcc(
                "-E",
                "-I",
                str(self.site_data_path / "includes"),
                "-o",
                "-",
                src_file,
                cwd=tmpdir,
                timeout=60,
                volumes={
                    str(self.site_data_path / "includes"): {
                        "bind": str(self.site_data_path / "includes"),
                        "mode": "ro",
                    }
                },
            )

        return self.benchmark_class.create(
            f"{self.name}/{seed}", preprocessed_src.encode("utf-8"), src
        )
