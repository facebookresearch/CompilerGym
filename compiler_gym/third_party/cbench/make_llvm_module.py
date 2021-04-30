#!/usr/bin/env python3
#
# This script compiles and links the sources for a cBench benchmark into a
# single unoptimized LLVM module.
#
# Usage:
#
#     $ make_cBench_llvm_module.py <in_dir> <outpath> [<cflag>...]
#
# This compiles the code from <in_dir> and generates an LLVM bitcode module at
# the given <outpath>, using any additional <cflags> as clang arguments.

import sys
from pathlib import Path
from typing import List

from compiler_gym.envs.llvm.llvm_benchmark import make_benchmark


def make_cbench_llvm_module(
    benchmark_dir: Path, cflags: List[str], output_path: Path
) -> str:
    """Compile a cBench benchmark into an unoptimized LLVM bitcode file."""
    src_dir = benchmark_dir / "src"
    if not src_dir.is_dir():
        src_dir = benchmark_dir
    assert src_dir.is_dir(), f"Source directory not found: {src_dir}"

    src_files = [path for path in src_dir.iterdir() if path.name.endswith(".c")]
    assert src_files, f"No source files in {src_dir}"

    benchmark = make_benchmark(inputs=src_files, copt=cflags or [])
    # Write just the bitcode to file.
    with open(output_path, "wb") as f:
        f.write(benchmark.proto.program.contents)


def main():
    """Main entry point."""
    # Parse arguments.
    benchmark_dir, output_path, *cflags = sys.argv[1:]
    benchmark_dir = Path(benchmark_dir).absolute().resolve()
    output_path = Path(output_path).absolute().resolve()

    make_cbench_llvm_module(benchmark_dir, cflags, output_path)


if __name__ == "__main__":
    main()
