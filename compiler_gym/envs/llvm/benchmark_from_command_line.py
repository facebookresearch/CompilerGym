# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import subprocess
import tempfile
import urllib.parse

from compiler_gym.datasets import Benchmark, BenchmarkInitError
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from compiler_gym.third_party.gccinvocation.gccinvocation import GccInvocation
from compiler_gym.util.commands import Popen
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.shell_format import join_cmd

logger = logging.getLogger(__name__)


class BenchmarkFromCommandLine(Benchmark):
    """A benchmark that has been constructed from a command line invocation.

    See :meth:`env.make_benchmark_from_command_line()
    <compiler_gym.envs.LlvmEnv.make_benchmark_from_command_line>`.
    """

    def __init__(self, invocation: GccInvocation, bitcode: bytes, timeout: int):
        uri = f"benchmark://clang-v0/{urllib.parse.quote_plus(join_cmd(invocation.original_argv))}"
        super().__init__(
            proto=BenchmarkProto(uri=str(uri), program=File(contents=bitcode))
        )
        self.command_line = invocation.original_argv

        # Modify the commandline so that it takes the bitcode file as input.
        #
        # Strip the original sources from the build command, but leave any
        # object file inputs.
        sources = set(s for s in invocation.sources if not s.endswith(".o"))
        build_command = [arg for arg in invocation.original_argv if arg not in sources]

        # Convert any object file inputs to absolute paths since the backend
        # service will have a different working directory.
        #
        # TODO(github.com/facebookresearch/CompilerGym/issues/325): To support
        # distributed execution, we should embed the contents of these object
        # files in the benchmark proto.
        object_files = set(s for s in invocation.sources if s.endswith(".o"))
        build_command = [
            os.path.abspath(arg) if arg in object_files else arg
            for arg in build_command
        ]

        # Append the new source to the build command and specify the absolute path
        # to the output.
        for i in range(len(build_command) - 2, -1, -1):
            if build_command[i] == "-o":
                del build_command[i + 1]
                del build_command[i]
        build_command += ["-xir", "$IN", "-o", str(invocation.output_path)]
        self.proto.dynamic_config.build_cmd.argument[:] = build_command
        self.proto.dynamic_config.build_cmd.outfile[:] = [str(invocation.output_path)]
        self.proto.dynamic_config.build_cmd.timeout_seconds = timeout

    def compile(self, env, timeout: int = 60) -> None:
        """This completes the compilation and linking of the final executable
        specified by the original command line.
        """
        with tempfile.NamedTemporaryFile(
            dir=transient_cache_path("."), prefix="benchmark-", suffix=".bc"
        ) as f:
            bitcode_path = f.name
            env.write_bitcode(bitcode_path)

            # Set the placeholder for input path.
            cmd = list(self.proto.dynamic_config.build_cmd.argument).copy()
            cmd = [bitcode_path if c == "$IN" else c for c in cmd]

            logger.debug(f"$ {join_cmd(cmd)}")

            with Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ) as lower:
                stdout, _ = lower.communicate(timeout=timeout)

            if lower.returncode:
                raise BenchmarkInitError(
                    f"Failed to lower LLVM bitcode with error:\n"
                    f"{stdout.decode('utf-8').rstrip()}\n"
                    f"Running command: {join_cmd(cmd)}"
                )
