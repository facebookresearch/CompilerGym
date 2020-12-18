#!/usr/bin/env python3
#
# This script compiles and links the sources for a cBench benchmark into a
# single unoptimized LLVM module.
#
# Usage:
#
#     $ make_cBench_llvm_module.py \
#         --benchmark=<benchmark_name> \
#         --cflags=<cflags> \
#         --output_path=<path_of_bitcode_file_to_write>
#
# The generated file embeds the LLVM-IR into a static const char* declaration,
# e.g.
#
#     static const char* CBENCH_DIJKSTRA_IR = R"_llvm_module_(
#     ...
#     )_llvm_module_";
#
# This can then be loaded into a buffer and passed to llvm::parseIR().

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from absl import flags

from compiler_gym.util.runfiles_path import runfiles_path

flags.DEFINE_string("path", None, "The name of the cBench benchamrk.")
flags.DEFINE_list("cflags", [], "A list of additional compiler flags.")
flags.DEFINE_string("output_path", None, "The output path.")
FLAGS = flags.FLAGS

# Path of the LLVM binaries.
CLANG = Path(runfiles_path("llvm/10.0.0/clang"))
LLVM_LINK = Path(runfiles_path("llvm/10.0.0/llvm-link"))


def make_cbench_llvm_module(
    benchmark_dir: Path, cflags: List[str], output_path: Path
) -> str:
    """Compile a cBench benchmark into an unoptimized LLVM bitcode file."""
    cflags = cflags or []

    src_dir = benchmark_dir / "src"
    assert src_dir.is_dir(), f"Source directory not found: {src_dir}"

    clang_command = [
        str(CLANG),
        "-w",
        "-emit-llvm",
        "-c",
        "-isystem",
        str(src_dir),
        # Defer optimizations but prevent clang from adding `optnone` function
        # annotations. See: https://bugs.llvm.org/show_bug.cgi?id=35950
        "-O0",
        "-Xclang",
        "-disable-O0-optnone",
        "-Xclang",
        "-disable-llvm-passes",
    ] + cflags

    # NOTE(cummins): The LLVM release does not include a full set of standard
    # includes. Hack around this for macOS.
    if Path(
        "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include"
    ).is_dir():
        clang_command += [
            "-isystem",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include",
            "-isystem",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/usr/include/machine",
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Build a list of all C sources.
        src_files = [path for path in src_dir.iterdir() if path.name.endswith(".c")]
        assert src_files, f"No source files in {src_dir}"

        ir_files = []
        # Compile a bitcode file for each C source.
        for path in src_files:
            ir_file = tmpdir / f"{len(ir_files)}.bc"
            subprocess.check_call(clang_command + [str(path), "-o", str(ir_file)])
            ir_files.append(str(ir_file))
        assert len(ir_files) == len(src_files)

        # Link the bitcode files to produce the LLVM-IR.
        output_path.parent.mkdir(exist_ok=True, parents=True)
        return subprocess.check_call(
            [str(LLVM_LINK), "-o", str(output_path)] + ir_files
        )


def main():
    """Main entry point."""
    assert CLANG.is_file(), f"clang not found: {CLANG}"
    assert LLVM_LINK.is_file(), f"llvm-link not found: {LLVM_LINK}"

    argv = FLAGS(sys.argv)
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"

    benchmark_dir = Path(FLAGS.path).absolute().resolve()
    cflags = FLAGS.cflags
    output_path = Path(FLAGS.output_path).absolute().resolve()

    make_cbench_llvm_module(benchmark_dir, cflags, output_path)


if __name__ == "__main__":
    main()
