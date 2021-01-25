# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a utility function for constructing LLVM benchmarks."""
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Union

from compiler_gym.service.proto import Benchmark, File
from compiler_gym.util.runfiles_path import cache_path, runfiles_path

CLANG = runfiles_path("CompilerGym/compiler_gym/third_party/llvm/clang")
LLVM_LINK = runfiles_path("CompilerGym/compiler_gym/third_party/llvm/llvm-link")


def _communicate(process, input=None, timeout=None):
    """subprocess.communicate() which kills subprocess on timeout."""
    try:
        return process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # kill() was added in Python 3.7.
        if sys.version_info >= (3, 7, 0):
            process.kill()
        else:
            process.terminate()
        raise


def _get_system_includes() -> Iterable[Path]:
    """Run the system compiler in verbose mode on a dummy input to get the
    system header search path.
    """
    system_compiler = os.environ.get("CXX", "c++")
    # Create a temporary directory to write the compiled 'binary' to, since
    # GNU assembler does not support piping to stdout.
    with tempfile.TemporaryDirectory() as d:
        process = subprocess.Popen(
            [system_compiler, "-xc++", "-v", "-c", "-", "-o", str(Path(d) / "a.out")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        _, stderr = _communicate(process, input="", timeout=30)
    if process.returncode:
        raise OSError(
            f"Failed to invoke {system_compiler}. "
            f"Is there a working system compiler?\n"
            f"Error: {stderr.strip()}"
        )

    in_search_list = False
    for line in stderr.split("\n"):
        if in_search_list and line.startswith("End of search list"):
            break
        elif in_search_list:
            yield Path(line.strip())
        elif line.startswith("#include <...> search starts here:"):
            in_search_list = True
    else:
        raise OSError(
            f"Failed to parse '#include <...>' search paths from {system_compiler}:\n"
            f"{stderr.strip()}"
        )


# Memoized search paths.
_SYSTEM_INCLUDES = None


def get_system_includes() -> List[Path]:
    """Determine the system include paths for C/C++ compilation jobs.

    This uses the system compiler to determine the search paths for C/C++ system
    headers. By default, :code:`c++` is invoked. This can be overridden by
    setting :code:`os.environ["CXX"]`.

    :return: A list of paths to system header directories.
    :raises OSError: If the compiler fails, or if the search paths cannot be
        determined.
    """
    # Memoize the system includes paths.
    global _SYSTEM_INCLUDES
    if _SYSTEM_INCLUDES is None:
        _SYSTEM_INCLUDES = list(_get_system_includes())
    return _SYSTEM_INCLUDES


class ClangInvocation(object):
    """Class to represent a single invocation of the clang compiler."""

    def __init__(
        self, args: List[str], system_includes: bool = True, timeout: int = 600
    ):
        """Create a clang invocation.

        :param args: The list of arguments to pass to clang.
        :param system_includes: Whether to include the system standard libraries
            during compilation jobs. This requires a system toolchain. See
            :func:`get_system_includes`.
        :param timeout: The maximum number of seconds to allow clang to run
            before terminating.
        """
        self.args = args
        self.system_includes = system_includes
        self.timeout = timeout

    def command(self, outpath: Path) -> List[str]:
        cmd = [str(CLANG)]
        if self.system_includes:
            for directory in get_system_includes():
                cmd += ["-isystem", str(directory)]

        cmd += [str(s) for s in self.args]
        cmd += ["-c", "-emit-llvm", "-o", str(outpath)]

        return cmd


def _run_command(args):
    cmd, timeout = args
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, universal_newlines=True
    )
    _, stderr = _communicate(process, timeout=timeout)
    if process.returncode:
        raise OSError(
            f"Compilation job failed with returncode {process.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {stderr.strip()}"
        )


def make_benchmark(
    inputs: Union[str, Path, ClangInvocation, List[Union[str, Path, ClangInvocation]]],
    copt: Optional[List[str]] = None,
    system_includes: bool = True,
    timeout: int = 600,
) -> Benchmark:
    """Create a benchmark for use by LLVM environments.

    This function takes one or more inputs and uses them to create a benchmark
    that can be passed to :meth:`compiler_gym.envs.LlvmEnv.reset`.

    For single-source C/C++ programs, you can pass the path of the source file:

    >>> benchmark = make_benchmark('my_app.c')
    >>> env = gym.make("llvm-v0")
    >>> env.reset(benchmark=benchmark)

    The clang invocation used is roughly equivalent to:

    .. code-block::

        $ clang my_app.c -O0 -c -emit-llvm -o benchmark.bc

    Additional compile-time arguments to clang can be provided using the
    :code:`copt` argument:

    >>> benchmark = make_benchmark('/path/to/my_app.cpp', copt=['-O2'])

    If you need more fine-grained control over the options, you can directly
    construct a :class:`ClangInvocation <compiler_gym.envs.llvm.ClangInvocation>`
    to pass a list of arguments to clang:

    >>> benchmark = make_benchmark(
        ClangInvocation(['/path/to/my_app.c'], timeout=10)
    )

    For multi-file programs, pass a list of inputs that will be compiled
    separately and then linked to a single module:

    >>> benchmark = make_benchmark([
        'main.c',
        'lib.cpp',
        'lib2.bc',
    ])

    If you already have prepared bitcode files, those can be linked and used
    directly:

    >>> benchmark = make_benchmark([
        'bitcode1.bc',
        'bitcode2.bc',
    ])

    .. note::
        LLVM bitcode compatibility is
        `not guaranteed <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_,
        so you must ensure that any precompiled bitcodes are compatible with the
        LLVM version used by CompilerGym, which can be queried using
        :func:`LlvmEnv.compiler_version <compiler_gym.envs.CompilerEnv.compiler_version>`.

    :param inputs: An input, or list of inputs.
    :param copt: A list of command line options to pass to clang when compiling
        source files.
    :param system_includes: Whether to include the system standard libraries
        during compilation jobs. This requires a system toolchain. See
        :func:`get_system_includes`.
    :param timeout: The maximum number of seconds to allow clang to run before
        terminating.
    :return: A :code:`Benchmark` message.
    :raises FileNotFoundError: If any input sources are not found.
    :raises TypeError: If the inputs are of unsupported types.
    :raises OSError: If a compilation job fails.
    :raises TimeoutExpired: If a compilation job exceeds :code:`timeout` seconds.
    """
    copt = copt or []

    bitcodes: List[Path] = []
    clang_jobs: List[ClangInvocation] = []

    def _add_path(path: Path):
        # NOTE(cummins): There is some discussion about the best way to create
        # a bitcode that is unoptimized yet does not hinder downstream
        # optimization opportunities. Here we are using a configuration based
        # on -O0, yet there is a suggestion that an optimized configuration
        # can produce better results if the optimizations themselves are
        # explicitly disabled, as in: ["-Oz", "-Xclang", "-disable-llvm-optzns"]
        # See: https://lists.llvm.org/pipermail/llvm-dev/2018-August/thread.html#125365
        DEFAULT_COPT = [
            "-O",
            "-Xclang",
            "-disable-O0-optnone",
            "-Xclang",
            "-disable-llvm-passes",
        ]

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.suffix == ".bc":
            bitcodes.append(path)
        elif path.suffix in {".c", ".cxx", ".cpp", ".cc"}:
            clang_jobs.append(
                ClangInvocation(
                    [str(path)] + DEFAULT_COPT + copt,
                    system_includes=system_includes,
                    timeout=timeout,
                )
            )
        else:
            raise ValueError(f"Unrecognized file type: {path.name}")

    # Determine from inputs the list of pre-compiled bitcodes and the clang
    # invocations required to compile the bitcodes.
    if isinstance(inputs, str) or isinstance(inputs, Path):
        _add_path(Path(inputs))
    elif isinstance(inputs, ClangInvocation):
        clang_jobs.append(inputs)
    else:
        for input in inputs:
            if isinstance(input, str) or isinstance(input, Path):
                _add_path(Path(input))
            elif isinstance(input, ClangInvocation):
                clang_jobs.append(input)
            else:
                raise TypeError(f"Invalid input type: {type(input).__name__}")

    if not bitcodes and not clang_jobs:
        raise ValueError("No inputs")

    # Shortcut if we only have a single pre-compiled bitcode.
    if len(bitcodes) == 1 and not clang_jobs:
        bitcode = bitcodes[0]
        return Benchmark(
            uri=f"file:///{bitcode}", program=File(uri=f"file:///{bitcode}")
        )

    with tempfile.TemporaryDirectory(dir=cache_path(".")) as d:
        working_dir = Path(d)

        # Run the clang invocations in parallel.
        clang_outs = [
            working_dir / f"out-{i}.bc" for i in range(1, len(clang_jobs) + 1)
        ]
        clang_cmds = [
            (job.command(out), job.timeout) for job, out in zip(clang_jobs, clang_outs)
        ]
        with multiprocessing.Pool() as pool:
            list(pool.imap_unordered(_run_command, clang_cmds))

        # Check that the expected files were generated.
        for i, b in enumerate(clang_outs):
            if not b.is_file():
                raise OSError(
                    f"Clang invocation failed to produce a file: {' '.join(clang_cmds[i])}"
                )

        if len(bitcodes + clang_outs) > 1:
            # Link all of the bitcodes into a single module.
            llvm_link_cmd = [str(LLVM_LINK), "-o", "-"] + [
                str(path) for path in bitcodes + clang_outs
            ]
            llvm_link = subprocess.Popen(
                llvm_link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            bitcode, stderr = _communicate(llvm_link, timeout=timeout)
            if llvm_link.returncode:
                raise OSError(
                    f"Failed to link LLVM bitcodes with error: {stderr.decode('utf-8')}"
                )
        else:
            # We only have a single bitcode so read it.
            with open(str(list(bitcodes + clang_outs)[0]), "rb") as f:
                bitcode = f.read()

    timestamp = datetime.now().strftime(f"%Y%m%HT%H%M%S-{random.randrange(16**4):04x}")
    return Benchmark(
        uri=f"benchmark://user/{timestamp}", program=File(contents=bitcode)
    )
