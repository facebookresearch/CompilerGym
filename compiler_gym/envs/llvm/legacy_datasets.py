# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the available LLVM datasets."""
import enum
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional

import fasteners

from compiler_gym.datasets.dataset import LegacyDataset
from compiler_gym.third_party import llvm
from compiler_gym.util import thread_pool
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, site_data_path
from compiler_gym.util.timer import Timer
from compiler_gym.validation_result import ValidationError

_CBENCH_DATA_URL = (
    "https://dl.fbaipublicfiles.com/compiler_gym/cBench-v0-runtime-data.tar.bz2"
)
_CBENCH_DATA_SHA256 = "a1b5b5d6b115e5809ccaefc2134434494271d184da67e2ee43d7f84d07329055"


if sys.platform == "darwin":
    _COMPILE_ARGS = [
        "-L",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
    ]
else:
    _COMPILE_ARGS = []

LLVM_DATASETS = [
    LegacyDataset(
        name="blas-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=300,
        size_bytes=3969036,
        sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
    ),
    # The difference between cBench-v0 and cBench-v1 is the arguments passed to
    # clang when preparing the LLVM bitcodes:
    #
    #   - v0: `-O0 -Xclang -disable-O0-optnone`.
    #   - v1: `-O1 -Xclang -Xclang -disable-llvm-passes`.
    #
    # The key difference with is that in v0, the generated IR functions were
    # annotated with a `noinline` attribute that prevented inline. In v1 that is
    # no longer the case.
    LegacyDataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-macos.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=7154448,
        sha256="072a730c86144a07bba948c49afe543e4f06351f1cb17f7de77f91d5c1a1b120",
        platforms=["macos"],
        deprecated_since="v0.1.4",
    ),
    LegacyDataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-linux.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=6940416,
        sha256="9b5838a90895579aab3b9375e8eeb3ed2ae58e0ad354fec7eb4f8b31ecb4a360",
        platforms=["linux"],
        deprecated_since="v0.1.4",
    ),
    LegacyDataset(
        name="cBench-v1",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v1-macos.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=10292032,
        sha256="90b312b40317d9ee9ed09b4b57d378879f05e8970bb6de80dc8581ad0e36c84f",
        platforms=["macos"],
    ),
    LegacyDataset(
        name="cBench-v1",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v1-linux.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=10075608,
        sha256="601fff3944c866f6617e653b6eb5c1521382c935f56ca1f36a9f5cf1a49f3de5",
        platforms=["linux"],
    ),
    LegacyDataset(
        name="github-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2",
        license="CC BY 4.0",
        description="https://zenodo.org/record/4122437",
        compiler="llvm-10.0.0",
        file_count=50708,
        size_bytes=725974100,
        sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
    ),
    LegacyDataset(
        name="linux-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2",
        license="GPL-2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=13920,
        size_bytes=516031044,
        sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
    ),
    LegacyDataset(
        name="mibench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=40,
        size_bytes=238480,
        sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
    ),
    LegacyDataset(
        name="npb-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2",
        license="NASA Open Source Agreement v1.3",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=122,
        size_bytes=2287444,
        sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
    ),
    LegacyDataset(
        name="opencv-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=442,
        size_bytes=21903008,
        sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
    ),
    LegacyDataset(
        name="poj104-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://sites.google.com/site/treebasedcnn/",
        compiler="llvm-10.0.0",
        file_count=49628,
        size_bytes=304207752,
        sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
    ),
    # FIXME(github.com/facebookresearch/CompilerGym/issues/55): Polybench
    # dataset has `optnone` function attribute set, requires rebuild.
    # LegacyDataset(
    #     name="polybench-v0",
    #     url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-polybench-v0.tar.bz2",
    #     license="BSD 3-Clause",
    #     description="https://github.com/ctuning/ctuning-programs",
    #     compiler="llvm-10.0.0",
    #     file_count=27,
    #     size_bytes=162624,
    #     sha256="968087e68470e5b44dc687dae195143000c7478a23d6631b27055bb3bb3116b1",
    # ),
    LegacyDataset(
        name="tensorflow-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-tensorflow-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=1985,
        size_bytes=299697312,
        sha256="f77dd1988c772e8359e1303cc9aba0d73d5eb27e0c98415ac3348076ab94efd1",
    ),
]


class BenchmarkExecutionResult(NamedTuple):
    """The result of running a benchmark."""

    walltime_seconds: float
    """The execution time in seconds."""

    error: Optional[ValidationError] = None
    """An error."""

    output: Optional[str] = None
    """The output generated by the benchmark."""

    def json(self):
        return self._asdict()


class LlvmSanitizer(enum.IntEnum):
    """The LLVM sanitizers."""

    ASAN = 1
    TSAN = 2
    MSAN = 3
    UBSAN = 4


# Compiler flags that are enabled by sanitizers.
_SANITIZER_FLAGS = {
    LlvmSanitizer.ASAN: ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"],
    LlvmSanitizer.TSAN: ["-O1", "-g", "-fsanitize=thread"],
    LlvmSanitizer.MSAN: ["-O1", "-g", "-fsanitize=memory"],
    LlvmSanitizer.UBSAN: ["-fsanitize=undefined"],
}


def _compile_and_run_bitcode_file(
    bitcode_file: Path,
    cmd: str,
    cwd: Path,
    linkopts: List[str],
    env: Dict[str, str],
    num_runs: int,
    logger: logging.Logger,
    sanitizer: Optional[LlvmSanitizer] = None,
    timeout_seconds: float = 300,
    compilation_timeout_seconds: float = 60,
) -> BenchmarkExecutionResult:
    """Run the given cBench benchmark."""
    # cBench benchmarks expect that a file _finfo_dataset exists in the
    # current working directory and contains the number of benchmark
    # iterations in it.
    with open(cwd / "_finfo_dataset", "w") as f:
        print(num_runs, file=f)

    # Create a barebones execution environment for the benchmark.
    run_env = {
        "TMPDIR": os.environ.get("TMPDIR", ""),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        # Disable all logging from GRPC. In the past I have had false-positive
        # "Wrong output" errors caused by GRPC error messages being logged to
        # stderr.
        "GRPC_VERBOSITY": "NONE",
    }
    run_env.update(env)

    error_data = {}

    if sanitizer:
        clang_path = llvm.clang_path()
        binary = cwd / "a.out"
        error_data["run_cmd"] = cmd.replace("$BIN", "./a.out")
        # Generate the a.out binary file.
        compile_cmd = (
            [clang_path.name, str(bitcode_file), "-o", str(binary)]
            + _COMPILE_ARGS
            + list(linkopts)
            + _SANITIZER_FLAGS.get(sanitizer, [])
        )
        error_data["compile_cmd"] = compile_cmd
        logger.debug("compile: %s", compile_cmd)
        assert not binary.is_file()
        clang = subprocess.Popen(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env={"PATH": f"{clang_path.parent}:{os.environ.get('PATH', '')}"},
        )
        try:
            output, _ = clang.communicate(timeout=compilation_timeout_seconds)
        except subprocess.TimeoutExpired:
            clang.kill()
            error_data["timeout"] = compilation_timeout_seconds
            return BenchmarkExecutionResult(
                walltime_seconds=timeout_seconds,
                error=ValidationError(
                    type="Compilation timeout",
                    data=error_data,
                ),
            )
        if clang.returncode:
            error_data["output"] = output
            return BenchmarkExecutionResult(
                walltime_seconds=timeout_seconds,
                error=ValidationError(
                    type="Compilation failed",
                    data=error_data,
                ),
            )
        assert binary.is_file()
    else:
        lli_path = llvm.lli_path()
        error_data["run_cmd"] = cmd.replace("$BIN", f"{lli_path.name} benchmark.bc")
        run_env["PATH"] = str(lli_path.parent)

    try:
        logger.debug("exec: %s", error_data["run_cmd"])
        with Timer() as timer:
            process = subprocess.Popen(
                error_data["run_cmd"],
                shell=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                env=run_env,
                cwd=cwd,
            )

            stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        error_data["timeout_seconds"] = timeout_seconds
        return BenchmarkExecutionResult(
            walltime_seconds=timeout_seconds,
            error=ValidationError(
                type="Execution timeout",
                data=error_data,
            ),
        )
    finally:
        if sanitizer:
            binary.unlink()

    try:
        output = stdout.decode("utf-8")
    except UnicodeDecodeError:
        output = "<binary>"

    if process.returncode:
        # Runtime error.
        if sanitizer == LlvmSanitizer.ASAN and "LeakSanitizer" in output:
            error_type = "Memory leak"
        elif sanitizer == LlvmSanitizer.ASAN and "AddressSanitizer" in output:
            error_type = "Memory error"
        elif sanitizer == LlvmSanitizer.MSAN and "MemorySanitizer" in output:
            error_type = "Memory error"
        elif "Segmentation fault" in output:
            error_type = "Segmentation fault"
        elif "Illegal Instruction" in output:
            error_type = "Illegal Instruction"
        else:
            error_type = f"Runtime error ({process.returncode})"

        error_data["return_code"] = process.returncode
        error_data["output"] = output
        return BenchmarkExecutionResult(
            walltime_seconds=timer.time,
            error=ValidationError(
                type=error_type,
                data=error_data,
            ),
        )
    return BenchmarkExecutionResult(walltime_seconds=timer.time, output=output)


def download_cBench_runtime_data() -> bool:
    """Download and unpack the cBench runtime dataset."""
    cbench_data = site_data_path("llvm/cBench-v1-runtime-data/runtime_data")
    if (cbench_data / "unpacked").is_file():
        return False
    else:
        # Clean up any partially-extracted data directory.
        if cbench_data.is_dir():
            shutil.rmtree(cbench_data)

        tar_contents = io.BytesIO(
            download(_CBENCH_DATA_URL, sha256=_CBENCH_DATA_SHA256)
        )
        with tarfile.open(fileobj=tar_contents, mode="r:bz2") as tar:
            cbench_data.parent.mkdir(parents=True, exist_ok=True)
            tar.extractall(cbench_data.parent)
        assert cbench_data.is_dir()
        # Create the marker file to indicate that the directory is unpacked
        # and ready to go.
        (cbench_data / "unpacked").touch()
        return True


# Thread lock to prevent race on download_cBench_runtime_data() from
# multi-threading. This works in tandem with the inter-process file lock - both
# are required.
_CBENCH_DOWNLOAD_THREAD_LOCK = Lock()


def _make_cBench_validator(
    cmd: str,
    linkopts: List[str],
    os_env: Dict[str, str],
    num_runs: int = 1,
    compare_output: bool = True,
    input_files: Optional[List[Path]] = None,
    output_files: Optional[List[Path]] = None,
    validate_result: Optional[
        Callable[[BenchmarkExecutionResult], Optional[str]]
    ] = None,
    pre_execution_callback: Optional[Callable[[Path], None]] = None,
    sanitizer: Optional[LlvmSanitizer] = None,
    flakiness: int = 5,
) -> Callable[["LlvmEnv"], Optional[ValidationError]]:  # noqa: F821
    """Construct a validation callback for a cBench benchmark. See validator() for usage."""
    input_files = input_files or []
    output_files = output_files or []

    def validator_cb(env: "LlvmEnv") -> Optional[ValidationError]:  # noqa: F821
        """The validation callback."""
        with _CBENCH_DOWNLOAD_THREAD_LOCK:
            with fasteners.InterProcessLock(cache_path("cBench-v1-runtime-data.LOCK")):
                download_cBench_runtime_data()

        cbench_data = site_data_path("llvm/cBench-v1-runtime-data/runtime_data")
        for input_file_name in input_files:
            path = cbench_data / input_file_name
            if not path.is_file():
                raise FileNotFoundError(f"Required benchmark input not found: {path}")

        # Create a temporary working directory to execute the benchmark in.
        with tempfile.TemporaryDirectory(dir=env.service.connection.working_dir) as d:
            cwd = Path(d)

            # Expand shell variable substitutions in the benchmark command.
            expanded_command = cmd.replace("$D", str(cbench_data))

            # Translate the output file names into paths inside the working
            # directory.
            output_paths = [cwd / o for o in output_files]

            if pre_execution_callback:
                pre_execution_callback(cwd)

            # Produce a gold-standard output using a reference version of
            # the benchmark.
            if compare_output or output_files:
                gs_env = env.fork()
                try:
                    # Reset to the original benchmark state and compile it.
                    gs_env.reset(benchmark=env.benchmark)
                    gs_env.write_bitcode(cwd / "benchmark.bc")
                    gold_standard = _compile_and_run_bitcode_file(
                        bitcode_file=cwd / "benchmark.bc",
                        cmd=expanded_command,
                        cwd=cwd,
                        num_runs=1,
                        # Use default optimizations for gold standard.
                        linkopts=linkopts + ["-O2"],
                        # Always assume safe.
                        sanitizer=None,
                        logger=env.logger,
                        env=os_env,
                    )
                    if gold_standard.error:
                        return ValidationError(
                            type=f"Gold standard: {gold_standard.error.type}",
                            data=gold_standard.error.data,
                        )
                finally:
                    gs_env.close()

                # Check that the reference run produced the expected output
                # files.
                for path in output_paths:
                    if not path.is_file():
                        try:
                            output = gold_standard.output
                        except UnicodeDecodeError:
                            output = "<binary>"
                        raise FileNotFoundError(
                            f"Expected file '{path.name}' not generated\n"
                            f"Benchmark: {env.benchmark}\n"
                            f"Command: {cmd}\n"
                            f"Output: {output}"
                        )
                    path.rename(f"{path}.gold_standard")

            # Serialize the benchmark to a bitcode file that will then be
            # compiled to a binary.
            env.write_bitcode(cwd / "benchmark.bc")
            outcome = _compile_and_run_bitcode_file(
                bitcode_file=cwd / "benchmark.bc",
                cmd=expanded_command,
                cwd=cwd,
                num_runs=num_runs,
                linkopts=linkopts,
                sanitizer=sanitizer,
                logger=env.logger,
                env=os_env,
            )

            if outcome.error:
                return outcome.error

            # Run a user-specified validation hook.
            if validate_result:
                validate_result(outcome)

            # Difftest the console output.
            if compare_output and gold_standard.output != outcome.output:
                return ValidationError(
                    type="Wrong output",
                    data={"expected": gold_standard.output, "actual": outcome.output},
                )

            # Difftest the output files.
            for i, path in enumerate(output_paths, start=1):
                if not path.is_file():
                    return ValidationError(
                        type="Output not generated",
                        data={"path": path.name, "command": cmd},
                    )
                diff = subprocess.Popen(
                    ["diff", str(path), f"{path}.gold_standard"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                stdout, _ = diff.communicate()
                if diff.returncode:
                    try:
                        stdout = stdout.decode("utf-8")
                        return ValidationError(
                            type="Wrong output (file)",
                            data={"path": path.name, "diff": stdout},
                        )
                    except UnicodeDecodeError:
                        return ValidationError(
                            type="Wrong output (file)",
                            data={"path": path.name, "diff": "<binary>"},
                        )

    def flaky_wrapped_cb(env: "LlvmEnv") -> Optional[ValidationError]:  # noqa: F821
        """Wrap the validation callback in a flakiness retry loop."""
        for i in range(1, max(flakiness, 1) + 1):
            try:
                error = validator_cb(env)
                if not error:
                    return
            except TimeoutError:
                # Timeout errors can be raised by the environment in case of a
                # slow step / observation, and should be retried.
                pass
            env.logger.warning(
                "Validation callback failed, attempt=%d/%d", i, flakiness
            )
        return error

    return flaky_wrapped_cb


# A map from benchmark name to validation callbacks. Defined below.
VALIDATORS: Dict[
    str, List[Callable[["LlvmEnv"], Optional[str]]]  # noqa: F821
] = defaultdict(list)


def validator(
    benchmark: str,
    cmd: str,
    data: Optional[List[str]] = None,
    outs: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
    compare_output: bool = True,
    validate_result: Optional[
        Callable[[BenchmarkExecutionResult], Optional[str]]
    ] = None,
    linkopts: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    pre_execution_callback: Optional[Callable[[], None]] = None,
    sanitizers: Optional[List[LlvmSanitizer]] = None,
) -> bool:
    """Declare a new benchmark validator.

    TODO(cummins): Pull this out into a public API.

    :param benchmark: The name of the benchmark that this validator supports.
    :cmd: The shell command to run the validation. Variable substitution is
        applied to this value as follows: :code:`$BIN` is replaced by the path
        of the compiled binary and :code:`$D` is replaced with the path to the
        benchmark's runtime data directory.
    :data: A list of paths to input files.
    :outs: A list of paths to output files.
    :return: :code:`True` if the new validator was registered, else :code:`False`.
    """
    platforms = platforms or ["linux", "macos"]
    if {"darwin": "macos"}.get(sys.platform, sys.platform) not in platforms:
        return False
    infiles = data or []
    outfiles = [Path(p) for p in outs or []]
    linkopts = linkopts or []
    env = env or {}
    if sanitizers is None:
        sanitizers = LlvmSanitizer

    VALIDATORS[benchmark].append(
        _make_cBench_validator(
            cmd=cmd,
            input_files=infiles,
            output_files=outfiles,
            compare_output=compare_output,
            validate_result=validate_result,
            linkopts=linkopts,
            os_env=env,
            pre_execution_callback=pre_execution_callback,
        )
    )

    # Register additional validators using the sanitizers.
    if sys.platform.startswith("linux"):
        for sanitizer in sanitizers:
            VALIDATORS[benchmark].append(
                _make_cBench_validator(
                    cmd=cmd,
                    input_files=infiles,
                    output_files=outfiles,
                    compare_output=compare_output,
                    validate_result=validate_result,
                    linkopts=linkopts,
                    os_env=env,
                    pre_execution_callback=pre_execution_callback,
                    sanitizer=sanitizer,
                )
            )

    return True


def get_llvm_benchmark_validation_callback(
    env: "LlvmEnv",  # noqa: F821
) -> Optional[Callable[["LlvmEnv"], Iterable[ValidationError]]]:  # noqa: F821
    """Return a callback for validating a given environment state.

    If there is no valid callback, returns :code:`None`.

    :param env: An :class:`LlvmEnv` instance.
    :return: An optional callback that takes an :class:`LlvmEnv` instance as
        argument and returns an optional string containing a validation error
        message.
    """
    validators = VALIDATORS.get(env.benchmark)

    # No match.
    if not validators:
        return None

    def composed(env):
        # Validation callbacks are read-only on the environment so it is
        # safe to run validators simultaneously in parallel threads.
        executor = thread_pool.get_thread_pool_executor()
        futures = (executor.submit(validator, env) for validator in validators)
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                yield result

        return None

    return composed


# ===============================
# Definition of cBench validators
# ===============================


def validate_sha_output(result: BenchmarkExecutionResult) -> Optional[str]:
    """SHA benchmark prints 5 random hex strings. Normally these hex strings are
    16 characters but occasionally they are less (presumably becuase of a
    leading zero being omitted).
    """
    try:
        if not re.match(
            r"[0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16}",
            result.output.rstrip(),
        ):
            return "Failed to parse hex output"
    except UnicodeDecodeError:
        return "Failed to parse unicode output"


def setup_ghostscript_library_files(dataset_id: int) -> Callable[[Path], None]:
    """Make a pre-execution setup hook for ghostscript."""

    def setup(cwd: Path):
        cbench_data = site_data_path("llvm/cBench-v1-runtime-data/runtime_data")
        # Copy the input data file into the current directory since ghostscript
        # doesn't like long input paths.
        shutil.copyfile(
            cbench_data / "office_data" / f"{dataset_id}.ps", cwd / "input.ps"
        )
        # Ghostscript doesn't like the library files being symlinks so copy them
        # into the working directory as regular files.
        for path in (cbench_data / "ghostscript").iterdir():
            if path.name.endswith(".ps"):
                shutil.copyfile(path, cwd / path.name)

    return setup


validator(
    benchmark="benchmark://cBench-v1/bitcount",
    cmd="$BIN 1125000",
)

validator(
    benchmark="benchmark://cBench-v1/bitcount",
    cmd="$BIN 512",
)

for i in range(1, 21):

    # NOTE(cummins): Disabled due to timeout errors, further investigation
    # needed.
    #
    # validator(
    #     benchmark="benchmark://cBench-v1/adpcm",
    #     cmd=f"$BIN $D/telecom_data/{i}.adpcm",
    #     data=[f"telecom_data/{i}.adpcm"],
    # )
    #
    # validator(
    #     benchmark="benchmark://cBench-v1/adpcm",
    #     cmd=f"$BIN $D/telecom_data/{i}.pcm",
    #     data=[f"telecom_data/{i}.pcm"],
    # )

    validator(
        benchmark="benchmark://cBench-v1/blowfish",
        cmd=f"$BIN d $D/office_data/{i}.benc output.txt 1234567890abcdeffedcba0987654321",
        data=[f"office_data/{i}.benc"],
        outs=["output.txt"],
    )

    validator(
        benchmark="benchmark://cBench-v1/bzip2",
        cmd=f"$BIN -d -k -f -c $D/bzip2_data/{i}.bz2",
        data=[f"bzip2_data/{i}.bz2"],
    )

    validator(
        benchmark="benchmark://cBench-v1/crc32",
        cmd=f"$BIN $D/telecom_data/{i}.pcm",
        data=[f"telecom_data/{i}.pcm"],
    )

    validator(
        benchmark="benchmark://cBench-v1/dijkstra",
        cmd=f"$BIN $D/network_dijkstra_data/{i}.dat",
        data=[f"network_dijkstra_data/{i}.dat"],
    )

    validator(
        benchmark="benchmark://cBench-v1/gsm",
        cmd=f"$BIN -fps -c $D/telecom_gsm_data/{i}.au",
        data=[f"telecom_gsm_data/{i}.au"],
    )

    # NOTE(cummins): ispell fails with returncode 1 and no output when run
    # under safe optimizations.
    #
    # validator(
    #     benchmark="benchmark://cBench-v1/ispell",
    #     cmd=f"$BIN -a -d americanmed+ $D/office_data/{i}.txt",
    #     data = [f"office_data/{i}.txt"],
    # )

    validator(
        benchmark="benchmark://cBench-v1/jpeg-c",
        cmd=f"$BIN -dct int -progressive -outfile output.jpeg $D/consumer_jpeg_data/{i}.ppm",
        data=[f"consumer_jpeg_data/{i}.ppm"],
        outs=["output.jpeg"],
        # NOTE(cummins): AddressSanitizer disabled because of
        # global-buffer-overflow in regular build.
        sanitizers=[LlvmSanitizer.TSAN, LlvmSanitizer.UBSAN],
    )

    validator(
        benchmark="benchmark://cBench-v1/jpeg-d",
        cmd=f"$BIN -dct int -outfile output.ppm $D/consumer_jpeg_data/{i}.jpg",
        data=[f"consumer_jpeg_data/{i}.jpg"],
        outs=["output.ppm"],
    )

    validator(
        benchmark="benchmark://cBench-v1/patricia",
        cmd=f"$BIN $D/network_patricia_data/{i}.udp",
        data=[f"network_patricia_data/{i}.udp"],
        env={
            # NOTE(cummins): Benchmark leaks when executed with safe optimizations.
            "ASAN_OPTIONS": "detect_leaks=0",
        },
    )

    validator(
        benchmark="benchmark://cBench-v1/qsort",
        cmd=f"$BIN $D/automotive_qsort_data/{i}.dat",
        data=[f"automotive_qsort_data/{i}.dat"],
        outs=["sorted_output.dat"],
        linkopts=["-lm"],
    )

    # NOTE(cummins): Rijndael benchmark disabled due to memory errors under
    # basic optimizations.
    #
    # validator(benchmark="benchmark://cBench-v1/rijndael", cmd=f"$BIN
    #     $D/office_data/{i}.enc output.dec d
    #     1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
    #     data=[f"office_data/{i}.enc"], outs=["output.dec"],
    # )
    #
    # validator(benchmark="benchmark://cBench-v1/rijndael", cmd=f"$BIN
    #     $D/office_data/{i}.txt output.enc e
    #     1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
    #     data=[f"office_data/{i}.txt"], outs=["output.enc"],
    # )

    validator(
        benchmark="benchmark://cBench-v1/sha",
        cmd=f"$BIN $D/office_data/{i}.txt",
        data=[f"office_data/{i}.txt"],
        compare_output=False,
        validate_result=validate_sha_output,
    )

    validator(
        benchmark="benchmark://cBench-v1/stringsearch",
        cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
        data=[f"office_data/{i}.txt"],
        outs=["output.txt"],
        env={
            # NOTE(cummins): Benchmark leaks when executed with safe optimizations.
            "ASAN_OPTIONS": "detect_leaks=0",
        },
        linkopts=["-lm"],
    )

    # NOTE(cummins): The stringsearch2 benchmark has a very long execution time.
    # Use only a single input to keep the validation time reasonable. I have
    # also observed Segmentation fault on gold standard using 4.txt and 6.txt.
    if i == 1:
        validator(
            benchmark="benchmark://cBench-v1/stringsearch2",
            cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
            data=[f"office_data/{i}.txt"],
            outs=["output.txt"],
            env={
                # NOTE(cummins): Benchmark leaks when executed with safe optimizations.
                "ASAN_OPTIONS": "detect_leaks=0",
            },
            # TSAN disabled because of extremely long execution leading to
            # timeouts.
            sanitizers=[LlvmSanitizer.ASAN, LlvmSanitizer.MSAN, LlvmSanitizer.UBSAN],
        )

    validator(
        benchmark="benchmark://cBench-v1/susan",
        cmd=f"$BIN $D/automotive_susan_data/{i}.pgm output_large.corners.pgm -c",
        data=[f"automotive_susan_data/{i}.pgm"],
        outs=["output_large.corners.pgm"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v1/tiff2bw",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
        data=[f"consumer_tiff_data/{i}.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
        env={
            # NOTE(cummins): Benchmark leaks when executed with safe optimizations.
            "ASAN_OPTIONS": "detect_leaks=0",
        },
    )

    validator(
        benchmark="benchmark://cBench-v1/tiff2rgba",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
        data=[f"consumer_tiff_data/{i}.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v1/tiffdither",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.bw.tif out.tif",
        data=[f"consumer_tiff_data/{i}.bw.tif"],
        outs=["out.tif"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v1/tiffmedian",
        cmd=f"$BIN $D/consumer_tiff_data/{i}.nocomp.tif output.tif",
        data=[f"consumer_tiff_data/{i}.nocomp.tif"],
        outs=["output.tif"],
        linkopts=["-lm"],
    )

    # NOTE(cummins): On macOS the following benchmarks abort with an illegal
    # hardware instruction error.
    # if sys.platform != "darwin":
    #     validator(
    #         benchmark="benchmark://cBench-v1/lame",
    #         cmd=f"$BIN $D/consumer_data/{i}.wav output.mp3",
    #         data=[f"consumer_data/{i}.wav"],
    #         outs=["output.mp3"],
    #         compare_output=False,
    #         linkopts=["-lm"],
    #     )

    # NOTE(cummins): Segfault on gold standard.
    #
    #     validator(
    #         benchmark="benchmark://cBench-v1/ghostscript",
    #         cmd="$BIN -sDEVICE=ppm -dNOPAUSE -dQUIET -sOutputFile=output.ppm -- input.ps",
    #         data=[f"office_data/{i}.ps"],
    #         outs=["output.ppm"],
    #         linkopts=["-lm", "-lz"],
    #         pre_execution_callback=setup_ghostscript_library_files(i),
    #     )
