# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a utility function for constructing LLVM benchmarks."""
import logging
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Union

from compiler_gym.datasets import Benchmark
from compiler_gym.errors import BenchmarkInitError
from compiler_gym.third_party import llvm
from compiler_gym.util.commands import Popen, communicate, run_command
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.shell_format import join_cmd
from compiler_gym.util.thread_pool import get_thread_pool_executor

logger = logging.getLogger(__name__)


class HostCompilerFailure(OSError):
    """Exception raised when the system compiler fails."""


class UnableToParseHostCompilerOutput(HostCompilerFailure):
    """Exception raised if unable to parse the verbose output of the host
    compiler."""


def _get_system_library_flags(compiler: str) -> Iterable[str]:
    """Private implementation function."""
    # Create a temporary file to write the compiled binary to, since GNU
    # assembler does not support piping to stdout.
    transient_cache = transient_cache_path(".")
    transient_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=transient_cache) as f:
        cmd = [compiler, "-xc++", "-v", "-", "-o", f.name]
        # On macOS we need to compile a binary to invoke the linker.
        if sys.platform != "darwin":
            cmd.append("-c")

        # Retry loop to permit timeouts, though unlikely, in case of a
        # heavily overloaded system (I have observed CI failures because
        # of this).
        for _ in range(3):
            try:
                with Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    universal_newlines=True,
                ) as process:
                    _, stderr = communicate(
                        process=process, input="int main(){return 0;}", timeout=30
                    )
                    if process.returncode:
                        raise HostCompilerFailure(
                            f"Failed to invoke '{compiler}'. "
                            f"Is there a working system compiler?\n"
                            f"Error: {stderr.strip()}"
                        )
                    break
            except subprocess.TimeoutExpired:
                continue
            except FileNotFoundError as e:
                raise HostCompilerFailure(
                    f"Failed to invoke '{compiler}'. "
                    f"Is there a working system compiler?\n"
                    f"Error: {e}"
                ) from e
        else:
            raise HostCompilerFailure(
                f"Compiler invocation '{join_cmd(cmd)}' timed out after 3 attempts."
            )

    # Parse the compiler output that matches the conventional output format
    # used by clang and GCC:
    #
    #     #include <...> search starts here:
    #     /path/1
    #     /path/2
    #     End of search list
    in_search_list = False
    lines = stderr.split("\n")
    for line in lines:
        if in_search_list and line.startswith("End of search list"):
            break
        elif in_search_list:
            # We have an include path to return.
            path = Path(line.strip())
            yield "-isystem"
            yield str(path)
            # Compatibility fix for compiling benchmark sources which use the
            # '#include <endian.h>' header, which on macOS is located in a
            # 'machine/endian.h' directory.
            if (path / "machine").is_dir():
                yield "-isystem"
                yield str(path / "machine")
        elif line.startswith("#include <...> search starts here:"):
            in_search_list = True
    else:
        msg = f"Failed to parse '#include <...>' search paths from '{compiler}'"
        stderr = stderr.strip()
        if stderr:
            msg += f":\n{stderr}"
        raise UnableToParseHostCompilerOutput(msg)

    if sys.platform == "darwin":
        yield "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"


@lru_cache(maxsize=32)
def _get_cached_system_library_flags(compiler: str) -> List[str]:
    """Private implementation detail."""
    return list(_get_system_library_flags(compiler))


def get_system_library_flags(compiler: Optional[str] = None) -> List[str]:
    """Determine the set of compilation flags needed to use the host system
    libraries.

    This uses the system compiler to determine the search paths for C/C++ system
    headers, and on macOS, the location of libclang_rt.osx.a. By default,
    :code:`c++` is invoked. This can be overridden by setting
    :code:`os.environ["CXX"]` prior to calling this function.

    :return: A list of command line flags for a compiler.

    :raises HostCompilerFailure: If the host compiler cannot be determined, or
        fails to compile a trivial piece of code.

    :raises UnableToParseHostCompilerOutput: If the output of the compiler
        cannot be understood.
    """
    compiler = compiler or (os.environ.get("CXX") or "c++")
    # We want to cache the results of this expensive query after resolving the
    # default value for the compiler argument, as it can changed based on
    # environment variables.
    return _get_cached_system_library_flags(compiler)


class ClangInvocation:
    """Class to represent a single invocation of the clang compiler."""

    def __init__(
        self, args: List[str], system_includes: bool = True, timeout: int = 600
    ):
        """Create a clang invocation.

        :param args: The list of arguments to pass to clang.
        :param system_includes: Whether to include the system standard libraries
            during compilation jobs. This requires a system toolchain. See
            :func:`get_system_library_flags`.
        :param timeout: The maximum number of seconds to allow clang to run
            before terminating.
        """
        self.args = args
        self.system_includes = system_includes
        self.timeout = timeout

    def command(self, outpath: Path) -> List[str]:
        cmd = [str(llvm.clang_path()), "-c", "-emit-llvm", "-o", str(outpath)]
        if self.system_includes:
            cmd += get_system_library_flags()
        cmd += [str(s) for s in self.args]

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


def make_benchmark(
    inputs: Union[str, Path, ClangInvocation, List[Union[str, Path, ClangInvocation]]],
    copt: Optional[List[str]] = None,
    system_includes: bool = True,
    timeout: int = 600,
) -> Benchmark:
    """Create a benchmark for use by LLVM environments.

    This function takes one or more inputs and uses them to create an LLVM
    bitcode benchmark that can be passed to
    :meth:`compiler_gym.envs.LlvmEnv.reset`.

    The following input types are supported:

    +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
    | **File Suffix**                                     | **Treated as**      | **Converted using**                                         |
    +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
    | :code:`.bc`                                         | LLVM IR bitcode     | No conversion required.                                     |
    +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
    | :code:`.ll`                                         | LLVM IR text format | Assembled to bitcode using llvm-as.                         |
    +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
    | :code:`.c`, :code:`.cc`, :code:`.cpp`, :code:`.cxx` | C / C++ source      | Compiled to bitcode using clang and the given :code:`copt`. |
    +-----------------------------------------------------+---------------------+-------------------------------------------------------------+

    .. note::

        The LLVM IR format has no compatability guarantees between versions (see
        `LLVM docs
        <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_).
        You must ensure that any :code:`.bc` and :code:`.ll` files are
        compatible with the LLVM version used by CompilerGym, which can be
        reported using :func:`env.compiler_version
        <compiler_gym.envs.CompilerEnv.compiler_version>`.

    E.g. for single-source C/C++ programs, you can pass the path of the source
    file:

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
            ClangInvocation(['/path/to/my_app.c'], system_includes=False, timeout=10)
        )

    For multi-file programs, pass a list of inputs that will be compiled
    separately and then linked to a single module:

        >>> benchmark = make_benchmark([
            'main.c',
            'lib.cpp',
            'lib2.bc',
            'foo/input.bc'
        ])

    :param inputs: An input, or list of inputs.

    :param copt: A list of command line options to pass to clang when compiling
        source files.

    :param system_includes: Whether to include the system standard libraries
        during compilation jobs. This requires a system toolchain. See
        :func:`get_system_library_flags`.

    :param timeout: The maximum number of seconds to allow clang to run before
        terminating.

    :return: A :code:`Benchmark` instance.

    :raises FileNotFoundError: If any input sources are not found.

    :raises TypeError: If the inputs are of unsupported types.

    :raises OSError: If a suitable compiler cannot be found.

    :raises BenchmarkInitError: If a compilation job fails.

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
            bitcodes.append(path.absolute())
        elif path.suffix in {".c", ".cc", ".cpp", ".cxx"}:
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
    if len(bitcodes) == 1 and not clang_jobs and not ll_paths:
        bitcode = bitcodes[0]
        return Benchmark.from_file(uri=f"benchmark://file-v0{bitcode}", path=bitcode)

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
                executor.submit(run_command, job.command(out), job.timeout)
                for job, out in zip(clang_jobs, clang_outs)
            ] + [
                executor.submit(run_command, command, timeout)
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
            with Popen(
                llvm_link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as llvm_link:
                bitcode, stderr = llvm_link.communicate(timeout=timeout)
                if llvm_link.returncode:
                    raise BenchmarkInitError(
                        f"Failed to link LLVM bitcodes with error: {stderr.decode('utf-8')}"
                    )

    timestamp = datetime.now().strftime("%Y%m%HT%H%M%S")
    uri = f"benchmark://user-v0/{timestamp}-{random.randrange(16**4):04x}"
    return Benchmark.from_file_contents(uri, bitcode)
