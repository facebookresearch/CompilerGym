# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from compiler_gym.datasets import Benchmark, BenchmarkSource, Dataset
from compiler_gym.datasets.benchmark import BenchmarkInitError, BenchmarkWithSource
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.envs.llvm import llvm_benchmark
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation
from compiler_gym.service.proto import BenchmarkDynamicConfig, Command
from compiler_gym.util.commands import Popen, communicate
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


class CsmithBenchmark(BenchmarkWithSource):
    """A CSmith benchmark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._src = None
        self.proto.dynamic_config.MergeFrom(
            BenchmarkDynamicConfig(
                build_cmd=Command(
                    argument=["$CC", "$IN"] + llvm_benchmark.get_system_library_flags(),
                    outfile=["a.out"],
                    timeout_seconds=60,
                ),
                run_cmd=Command(
                    argument=["./a.out"],
                    timeout_seconds=300,
                ),
            )
        )

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
        self.csmith_bin_path = csmith_bin or _CSMITH_BIN
        self.csmith_includes_path = csmith_includes or _CSMITH_INCLUDES
        # The command that is used to compile an LLVM-IR bitcode file from a
        # Csmith input. Reads from stdin, writes to stdout.
        self.clang_compile_command: List[str] = ClangInvocation.from_c_file(
            "-",  # Read from stdin.
            copt=[
                "-xc",  # The C programming language.
                "-ferror-limit=1",  # Stop on first error.
                "-w",  # No warnings.
                f"-I{self.csmith_includes_path}",  # Include the Csmith headers.
            ],
        ).command(
            outpath="-"  # Write to stdout.
        )

    @property
    def size(self) -> int:
        # Actually 2^32 - 1, but practically infinite for all intents and
        # purposes.
        return 0

    def benchmark_uris(self) -> Iterable[str]:
        return (f"{self.name}/{i}" for i in range(UINT_MAX))

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> CsmithBenchmark:
        seed = int(uri.path[1:])
        return self.benchmark_from_seed(seed)

    def _random_benchmark(self, random_state: np.random.Generator) -> Benchmark:
        seed = random_state.integers(UINT_MAX)
        return self.benchmark_from_seed(seed)

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
        try:
            with Popen(
                [str(self.csmith_bin_path), "--seed", str(seed)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as csmith:
                # Generate the C source.
                src, stderr = communicate(csmith, timeout=300)
                if csmith.returncode:
                    try:
                        stderr = "\n".join(
                            truncate(
                                stderr.decode("utf-8"), max_line_len=200, max_lines=20
                            )
                        )
                        logger.warning("Csmith failed with seed %d: %s", seed, stderr)
                    except UnicodeDecodeError:
                        # Failed to interpret the stderr output, generate a generic
                        # error message.
                        logger.warning("Csmith failed with seed %d", seed)
                    return self.benchmark_from_seed(
                        seed, max_retries=max_retries, retry_count=retry_count + 1
                    )

            # Compile to IR.
            with Popen(
                self.clang_compile_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ) as clang:
                stdout, _ = communicate(clang, input=src, timeout=300)
                if clang.returncode:
                    compile_cmd = " ".join(self.clang_compile_command)
                    raise BenchmarkInitError(
                        f"Compilation job failed!\n"
                        f"Csmith seed: {seed}\n"
                        f"Command: {compile_cmd}\n"
                    )
        except subprocess.TimeoutExpired:
            raise BenchmarkInitError(
                f"Benchmark generation using seed {seed} timed out"
            )

        return self.benchmark_class.create(f"{self.name}/{seed}", stdout, src)
