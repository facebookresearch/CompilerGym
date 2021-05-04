# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a utility function for constructing LLVM benchmarks."""
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from signal import Signals
from typing import Iterable, List, Optional, Union

from compiler_gym.datasets import Benchmark, BenchmarkInitError
from compiler_gym.third_party import llvm
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.thread_pool import get_thread_pool_executor


def _communicate(process, input=None, timeout=None):
    """subprocess.communicate() which kills subprocess on timeout."""
    try:
        return process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
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

    # Parse the compiler output that matches the conventional output format
    # used by clang and GCC:
    #
    #     #include <...> search starts here:
    #     /path/1
    #     /path/2
    #     End of search list
    in_search_list = False
    for line in stderr.split("\n"):
        if in_search_list and line.startswith("End of search list"):
            break
        elif in_search_list:
            # We have an include path to return.
            path = Path(line.strip())
            yield path
            # Compatibility fix for compiling benchmark sources which use the
            # '#include <endian.h>' header, which on macOS is located in a
            # 'machine/endian.h' directory.
            if (path / "machine").is_dir():
                yield path / "machine"
        elif line.startswith("#include <...> search starts here:"):
            in_search_list = True
    else:
        raise OSError(
            f"Failed to parse '#include <...>' search paths from {system_compiler}:\n"
            f"{stderr.strip()}"
        )


# Memoized search paths. Call get_system_includes() to access them.
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


class ClangInvocation:
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
        cmd = [str(llvm.clang_path())]
        if self.system_includes:
            for directory in get_system_includes():
                cmd += ["-isystem", str(directory)]

        cmd += [str(s) for s in self.args]
        cmd += ["-c", "-emit-llvm", "-o", str(outpath)]

        return cmd

    @classmethod
    def from_c_file(
        cls,
        path: Path,
        copt: Optional[List[str]] = None,
        system_includes: bool = True,
        timeout: int = 600,
    ) -> "ClangInvocation":
        copt = copt or []
        # NOTE(cummins): There is some discussion about the best way to create a
        # bitcode that is unoptimized yet does not hinder downstream
        # optimization opportunities. Here we are using a configuration based on
        # -O1 in which we prevent the -O1 optimization passes from running. This
        # is because LLVM produces different function attributes dependening on
        # the optimization level. E.g. "-O0 -Xclang -disable-llvm-optzns -Xclang
        # -disable-O0-optnone" will generate code with "noinline" attributes set
        # on the functions, wheras "-Oz -Xclang -disable-llvm-optzns" will
        # generate functions with "minsize" and "optsize" attributes set.
        #
        # See also:
        #   <https://lists.llvm.org/pipermail/llvm-dev/2018-August/thread.html#125365>
        #   <https://github.com/facebookresearch/CompilerGym/issues/110>
        DEFAULT_COPT = [
            "-O1",
            "-Xclang",
            "-disable-llvm-passes",
            "-Xclang",
            "-disable-llvm-optzns",
        ]

        return cls(
            DEFAULT_COPT + copt + [str(path)],
            system_includes=system_includes,
            timeout=timeout,
        )


def _run_command(cmd: List[str], timeout: int):
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, universal_newlines=True
    )
    _, stderr = _communicate(process, timeout=timeout)
    if process.returncode:
        returncode = process.returncode
        try:
            # Try and decode the name of a signal. Signal returncodes
            # are negative.
            returncode = f"{returncode} ({Signals(abs(returncode)).name})"
        except ValueError:
            pass
        raise BenchmarkInitError(
            f"Compilation job failed with returncode {returncode}\n"
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
    construct a :class:`ClangInvocation
    <compiler_gym.envs.llvm.ClangInvocation>` to pass a list of arguments to
    clang:

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

    Text-format LLVM assembly can also be used:

        >>> benchmark = make_benchmark('module.ll')

    .. note::

        LLVM bitcode compatibility is
        `not guaranteed <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_,
        so you must ensure that any precompiled bitcodes are compatible with the
        LLVM version used by CompilerGym, which can be queried using
        :func:`env.compiler_version <compiler_gym.envs.CompilerEnv.compiler_version>`.

    :param inputs: An input, or list of inputs.

    :param copt: A list of command line options to pass to clang when compiling
        source files.

    :param system_includes: Whether to include the system standard libraries
        during compilation jobs. This requires a system toolchain. See
        :func:`get_system_includes`.

    :param timeout: The maximum number of seconds to allow clang to run before
        terminating.

    :return: A :code:`Benchmark` instance.

    :raises FileNotFoundError: If any input sources are not found.

    :raises TypeError: If the inputs are of unsupported types.

    :raises OSError: If a compilation job fails.

    :raises TimeoutExpired: If a compilation job exceeds :code:`timeout`
        seconds.
    """
    copt = copt or []

    bitcodes: List[Path] = []
    clang_jobs: List[ClangInvocation] = []
    ll_paths: List[Path] = []

    def _add_path(path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)

        if path.suffix == ".bc":
            bitcodes.append(path)
        elif path.suffix in {".c", ".cxx", ".cpp", ".cc"}:
            clang_jobs.append(
                ClangInvocation.from_c_file(
                    path, copt=copt, system_includes=system_includes, timeout=timeout
                )
            )
        elif path.suffix == ".ll":
            ll_paths.append(path)
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

    # Shortcut if we only have a single pre-compiled bitcode.
    if len(bitcodes) == 1 and not clang_jobs:
        bitcode = bitcodes[0]
        return Benchmark.from_file(uri=f"file:///{bitcode}", path=bitcode)

    tmpdir_root = transient_cache_path(".")
    tmpdir_root.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(
        dir=tmpdir_root, prefix="llvm-make_benchmark-"
    ) as d:
        working_dir = Path(d)

        clang_outs = [
            working_dir / f"clang-out-{i}.bc" for i in range(1, len(clang_jobs) + 1)
        ]
        llvm_as_outs = [
            working_dir / f"llvm-as-out-{i}.bc" for i in range(1, len(ll_paths) + 1)
        ]

        # Run the clang and llvm-as invocations in parallel. Avoid running this
        # code path if possible as get_thread_pool_executor() requires locking.
        if clang_jobs or ll_paths:
            llvm_as_path = str(llvm.llvm_as_path())
            executor = get_thread_pool_executor()

            llvm_as_commands = [
                [llvm_as_path, str(ll_path), "-o", bc_path]
                for ll_path, bc_path in zip(ll_paths, llvm_as_outs)
            ]

            # Fire off the clang and llvm-as jobs.
            futures = [
                executor.submit(_run_command, job.command(out), job.timeout)
                for job, out in zip(clang_jobs, clang_outs)
            ] + [
                executor.submit(_run_command, command, timeout)
                for command in llvm_as_commands
            ]

            # Block until finished.
            list(future.result() for future in as_completed(futures))

            # Check that the expected files were generated.
            for clang_job, bc_path in zip(clang_jobs, clang_outs):
                if not bc_path.is_file():
                    raise BenchmarkInitError(
                        f"clang failed: {' '.join(clang_job.command(bc_path))}"
                    )
            for command, bc_path in zip(llvm_as_commands, llvm_as_outs):
                if not bc_path.is_file():
                    raise BenchmarkInitError(f"llvm-as failed: {command}")

        all_outs = bitcodes + clang_outs + llvm_as_outs
        if not all_outs:
            raise ValueError("No inputs")
        elif len(all_outs) == 1:
            # We only have a single bitcode so read it.
            with open(str(all_outs[0]), "rb") as f:
                bitcode = f.read()
        else:
            # Link all of the bitcodes into a single module.
            llvm_link_cmd = [str(llvm.llvm_link_path()), "-o", "-"] + [
                str(path) for path in bitcodes + clang_outs
            ]
            llvm_link = subprocess.Popen(
                llvm_link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            bitcode, stderr = _communicate(llvm_link, timeout=timeout)
            if llvm_link.returncode:
                raise BenchmarkInitError(
                    f"Failed to link LLVM bitcodes with error: {stderr.decode('utf-8')}"
                )

    timestamp = datetime.now().strftime("%Y%m%HT%H%M%S")
    uri = f"benchmark://user/{timestamp}-{random.randrange(16**4):04x}"
    return Benchmark.from_file_contents(uri, bitcode)
