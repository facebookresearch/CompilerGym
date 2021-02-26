# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the available LLVM datasets."""
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

import fasteners

from compiler_gym.datasets.dataset import Dataset
from compiler_gym.util.download import download
from compiler_gym.util.runfiles_path import cache_path, runfiles_path, site_data_path
from compiler_gym.util.timer import Timer

_CLANG = runfiles_path("compiler_gym/third_party/llvm/clang")

_CBENCH_DATA = site_data_path("llvm/cBench-v0-runtime-data/runtime_data")
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
    Dataset(
        name="blas-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-blas-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=300,
        size_bytes=3969036,
        sha256="e724a8114709f8480adeb9873d48e426e8d9444b00cddce48e342b9f0f2b096d",
    ),
    Dataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-macos.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=7154448,
        sha256="072a730c86144a07bba948c49afe543e4f06351f1cb17f7de77f91d5c1a1b120",
        platforms=["macos"],
    ),
    Dataset(
        name="cBench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-cBench-v0-linux.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=23,
        size_bytes=6940416,
        sha256="9b5838a90895579aab3b9375e8eeb3ed2ae58e0ad354fec7eb4f8b31ecb4a360",
        platforms=["linux"],
    ),
    Dataset(
        name="github-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-github-v0.tar.bz2",
        license="CC BY 4.0",
        description="https://zenodo.org/record/4122437",
        compiler="llvm-10.0.0",
        file_count=50708,
        size_bytes=725974100,
        sha256="880269dd7a5c2508ea222a2e54c318c38c8090eb105c0a87c595e9dd31720764",
    ),
    Dataset(
        name="linux-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-linux-v0.tar.bz2",
        license="GPL-2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=13920,
        size_bytes=516031044,
        sha256="a1ae5c376af30ab042c9e54dc432f89ce75f9ebaee953bc19c08aff070f12566",
    ),
    Dataset(
        name="mibench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-mibench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=40,
        size_bytes=238480,
        sha256="128c090c40b955b99fdf766da167a5f642018fb35c16a1d082f63be2e977eb13",
    ),
    Dataset(
        name="npb-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-npb-v0.tar.bz2",
        license="NASA Open Source Agreement v1.3",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=122,
        size_bytes=2287444,
        sha256="793ac2e7a4f4ed83709e8a270371e65b724da09eaa0095c52e7f4209f63bb1f2",
    ),
    Dataset(
        name="opencv-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-opencv-v0.tar.bz2",
        license="Apache 2.0",
        description="https://github.com/spcl/ncc/tree/master/data",
        compiler="llvm-10.0.0",
        file_count=442,
        size_bytes=21903008,
        sha256="003df853bd58df93572862ca2f934c7b129db2a3573bcae69a2e59431037205c",
    ),
    Dataset(
        name="poj104-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-poj104-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://sites.google.com/site/treebasedcnn/",
        compiler="llvm-10.0.0",
        file_count=49628,
        size_bytes=304207752,
        sha256="6254d629887f6b51efc1177788b0ce37339d5f3456fb8784415ed3b8c25cce27",
    ),
    Dataset(
        name="polybench-v0",
        url="https://dl.fbaipublicfiles.com/compiler_gym/llvm_bitcodes-10.0.0-polybench-v0.tar.bz2",
        license="BSD 3-Clause",
        description="https://github.com/ctuning/ctuning-programs",
        compiler="llvm-10.0.0",
        file_count=27,
        size_bytes=162624,
        sha256="968087e68470e5b44dc687dae195143000c7478a23d6631b27055bb3bb3116b1",
    ),
    Dataset(
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

    error: Optional[str] = None
    """An error message."""

    output: Optional[str] = None
    """The output generated by the benchmark."""


def _compile_and_run_bitcode_file(
    bitcode_file: Path,
    cmd: str,
    cwd: Path,
    linkopts: List[str],
    num_runs: int,
    timeout_seconds: float = 300,
) -> BenchmarkExecutionResult:
    """Run the given cBench benchmark."""
    binary = cwd / "a.out"

    # cBench benchmarks expect that a file _finfo_dataset exists in the
    # current working directory and contains the number of benchmark
    # iterations in it.
    with open(cwd / "_finfo_dataset", "w") as f:
        print(num_runs, file=f)

    # Generate the a.out binary file.
    assert not binary.is_file()
    clang = subprocess.Popen(
        [_CLANG, str(bitcode_file), "-o", str(binary)] + _COMPILE_ARGS + list(linkopts),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    output, _ = clang.communicate()
    if clang.returncode:
        return BenchmarkExecutionResult(
            walltime_seconds=timeout_seconds,
            error=output,
        )
    assert binary.is_file()

    # Create a barebones execution environment for the benchmark.
    env = {
        "TMPDIR": os.environ.get("TMPDIR", ""),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
    }

    try:
        with Timer() as timer:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return BenchmarkExecutionResult(
            walltime_seconds=timeout_seconds,
            error=f"Benchmark failed to complete within {timeout_seconds} timeout.",
        )
    finally:
        binary.unlink()

    if process.returncode:
        try:
            output = stdout.decode("utf-8")
            msg = f"Benchmark exited with returncode {process.returncode}. Output: {output}"
        except UnicodeDecodeError:
            msg = f"Benchmark exited with returncode {process.returncode}"
        return BenchmarkExecutionResult(walltime_seconds=timer.time, error=msg)

    return BenchmarkExecutionResult(walltime_seconds=timer.time, output=stdout)


@fasteners.interprocess_locked(cache_path("cBench-v0-runtime-data.LOCK"))
def download_cBench_runtime_data() -> bool:
    """Download and unpack the cBench runtime dataset."""
    if _CBENCH_DATA.is_dir():
        return False
    else:
        tar_contents = io.BytesIO(
            download(_CBENCH_DATA_URL, sha256=_CBENCH_DATA_SHA256)
        )
        with tarfile.open(fileobj=tar_contents, mode="r:bz2") as tar:
            _CBENCH_DATA.parent.mkdir(parents=True)
            tar.extractall(_CBENCH_DATA.parent)
        assert _CBENCH_DATA.is_dir()
        return True


def _make_cBench_validator(
    cmd: str,
    linkopts: List[str],
    num_runs: int = 1,
    compare_output: bool = True,
    input_files: Optional[List[Path]] = None,
    output_files: Optional[List[Path]] = None,
    validate_result: Optional[
        Callable[[BenchmarkExecutionResult], Optional[str]]
    ] = None,
    pre_execution_callback: Optional[Callable[[Path], None]] = None,
):
    """Construct a validation callback for a cBench benchmark. See validator() for usage."""
    input_files = input_files or []
    output_files = output_files or []

    def validator_cb(env):
        """The validation callback."""
        for path in input_files:
            if not path.is_file():
                raise FileNotFoundError(f"Required benchmark input not found: {path}")

        # Expand shell variable substitutions in the benchmark command.
        expanded_command = cmd.replace("$BIN", "./a.out").replace(
            "$D", str(_CBENCH_DATA)
        )

        # Create a temporary working directory to execute the benchmark in.
        with tempfile.TemporaryDirectory(dir=env.service.connection.working_dir) as d:
            cwd = Path(d)
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
                        linkopts=linkopts,
                    )
                    if gold_standard.error:
                        raise OSError(
                            f"Failed to produce reference output for benchmark '{env.benchmark}' "
                            f"using '{cmd}': {gold_standard.error}"
                        )
                finally:
                    gs_env.close()

                # Check that the reference run produced the expected output
                # files.
                for path in output_paths:
                    if not path.is_file():
                        try:
                            output = gold_standard.output.decode("utf-8")
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
            )

            if outcome.error:
                return outcome.error

            # Run a user-specified validation hook.
            if validate_result:
                validate_result(outcome)

            # Difftest the console output.
            if compare_output and gold_standard.output != outcome.output:
                try:
                    return (
                        f"Benchmark output differs from expected.\n"
                        f"Expected: {gold_standard.output.decode('utf-8')}\n"
                        f"Actual: {outcome.output.decode('utf-8')}"
                    )
                except UnicodeDecodeError:
                    return "Benchmark output differs from expected (binary diff)"

            # Difftest the output files.
            for path in output_paths:
                if not path.is_file():
                    return f"Expected file '{path.name}' not generated using command: {cmd}"
                diff = subprocess.Popen(
                    ["diff", str(path), f"{path}.gold_standard"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                stdout, _ = diff.communicate()
                if diff.returncode:
                    try:
                        stdout = stdout.decode("utf-8")
                        if stdout.startswith("Binary files "):
                            return f"Benchmark output file '{path.name}' differs from expected (binary diff)"
                        else:
                            return f"Benchmark output file '{path.name}' differs from expected: {stdout}"
                    except UnicodeDecodeError:
                        return f"Benchmark output file '{path.name}' differs from (binary diff)"

    return validator_cb


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
    linkopts: List[str] = None,
    pre_execution_callback: Optional[Callable[[], None]] = None,
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
    infiles = [_CBENCH_DATA / p for p in data or []]
    outfiles = [Path(p) for p in outs or []]
    linkopts = linkopts or []

    VALIDATORS[benchmark].append(
        _make_cBench_validator(
            cmd=cmd,
            input_files=infiles,
            output_files=outfiles,
            compare_output=compare_output,
            validate_result=validate_result,
            linkopts=linkopts,
            pre_execution_callback=pre_execution_callback,
        )
    )

    return True


def get_llvm_benchmark_validation_callback(
    env: "LlvmEnv",  # noqa: F821
) -> Optional[Callable[["LlvmEnv"], Optional[str]]]:  # noqa: F821
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
        download_cBench_runtime_data()

        # Validation callbacks are read-only on the environment so it is
        # safe to run validators simultaneously in parallel threads.
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = (executor.submit(validator, env) for validator in validators)
            results = (future.result() for future in as_completed(futures))
            errors = [result for result in results if result is not None]

            msg = []
            if errors:
                freq = sorted(Counter(errors).items(), key=lambda x: -x[1])
                for j, (message, count) in enumerate(freq, start=1):
                    if count > 1:
                        msg.append(f"[{j}/{len(freq)}] {count}× {message}")
                    else:
                        msg.append(f"[{j}/{len(freq)}] {message}")
                    if j >= 3:
                        msg.append("...")
                        break
                return (
                    f"Failed {len(errors)} of {len(validators)} validators: "
                    + "\n".join(msg)
                )

            return None

    return composed


# ===============================
# Definition of cBench validators
# ===============================


def validate_sha_output(result: BenchmarkExecutionResult):
    """SHA benchmark prints 5 random hex strings. Normally these hex strings are
    16 characters but occasionally they are less (presumably becuase of a
    leading zero being omitted).
    """
    assert re.match(
        r"[0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16} [0-9a-f]{0,16}",
        result.output.decode("utf-8").rstrip(),
    )


def setup_ghostscript_library_files(cwd: Path):
    """Pre-execution setup hook for ghostscript."""
    # Copy input data into current directory since ghostscript doesn't like long
    # input paths.
    for path in (_CBENCH_DATA / "office_data").iterdir():
        if path.name.endswith(".ps"):
            shutil.copyfile(path, cwd / path.name)
    # Ghostscript doesn't like the library files being symlinks so copy them
    # into the working directory as regular files.
    for path in (_CBENCH_DATA / "ghostscript").iterdir():
        if path.name.endswith(".ps"):
            shutil.copyfile(path, cwd / path.name)


validator(
    benchmark="benchmark://cBench-v0/bitcount",
    cmd="$BIN 1125000",
)

validator(
    benchmark="benchmark://cBench-v0/bitcount",
    cmd="$BIN 512",
)

for i in range(1, 21):
    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/adpcm",
    #     cmd=f"$BIN $D/telecom_data/{i}.adpcm",
    #     data=[f"telecom_data/{i}.adpcm"],
    # )

    # validator(
    #     benchmark="benchmark://cBench-v0/adpcm",
    #     cmd=f"$BIN $D/telecom_data/{i}.pcm",
    #     data=[f"telecom_data/{i}.pcm"],
    # )

    validator(
        benchmark="benchmark://cBench-v0/blowfish",
        cmd=f"$BIN d $D/office_data/{i}.benc output.txt 1234567890abcdeffedcba0987654321",
        data=[f"office_data/{i}.benc"],
        outs=["output.txt"],
    )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/bzip2",
    #     cmd=f"$BIN -d -k -f -c $D/bzip2_data/{i}.bz2",
    #     data=[f"bzip2_data/{i}.bz2"],
    # )

    validator(
        benchmark="benchmark://cBench-v0/crc32",
        cmd=f"$BIN $D/telecom_data/{i}.pcm",
        data=[f"telecom_data/{i}.pcm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/dijkstra",
        cmd=f"$BIN $D/network_dijkstra_data/{i}.dat",
        data=[f"network_dijkstra_data/{i}.dat"],
    )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/gsm",
    #     cmd=f"$BIN -fps -c $D/telecom_gsm_data/{i}.au",
    #     data=[f"telecom_gsm_data/{i}.au"],
    # )

    # TODO(cummins): ispell executable appears broken. Investigation needed.
    # validator(
    #     benchmark="benchmark://cBench-v0/ispell",
    #     cmd=f"$BIN -a -d americanmed+ $D/office_data/{i}.txt",
    #     data = [f"office_data/{i}.txt"],
    # )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/jpeg-c",
    #     cmd=f"$BIN -dct int -progressive -outfile output.jpeg $D/consumer_jpeg_data/{i}.ppm",
    #     data=[f"consumer_jpeg_data/{i}.ppm"],
    #     outs=["output.jpeg"],
    # )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/jpeg-d",
    #     cmd=f"$BIN -dct int -outfile output.ppm $D/consumer_jpeg_data/{i}.jpg",
    #     data=[f"consumer_jpeg_data/{i}.jpg"],
    #     outs=["output.ppm"],
    # )

    validator(
        benchmark="benchmark://cBench-v0/patricia",
        cmd=f"$BIN $D/network_patricia_data/{i}.udp",
        data=[f"network_patricia_data/{i}.udp"],
    )

    validator(
        benchmark="benchmark://cBench-v0/qsort",
        cmd=f"$BIN $D/automotive_qsort_data/{i}.dat",
        data=[f"automotive_qsort_data/{i}.dat"],
        outs=["sorted_output.dat"],
        linkopts=["-lm"],
    )

    validator(
        benchmark="benchmark://cBench-v0/rijndael",
        cmd=f"$BIN $D/office_data/{i}.enc output.dec d 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321",
        data=[f"office_data/{i}.enc"],
        outs=["output.dec"],
    )

    validator(
        benchmark="benchmark://cBench-v0/sha",
        cmd=f"$BIN $D/office_data/{i}.txt",
        data=[f"office_data/{i}.txt"],
        compare_output=False,
        validate_result=validate_sha_output,
    )

    validator(
        benchmark="benchmark://cBench-v0/stringsearch",
        cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
        data=[f"office_data/{i}.txt"],
        outs=["output.txt"],
        linkopts=["-lm"],
    )

    # TODO(cummins): Sporadic segfaults.
    # validator(
    #     benchmark="benchmark://cBench-v0/stringsearch2",
    #     cmd=f"$BIN $D/office_data/{i}.txt $D/office_data/{i}.s.txt output.txt",
    #     data=[f"office_data/{i}.txt"],
    #     outs=["output.txt"],
    # )

    validator(
        benchmark="benchmark://cBench-v0/susan",
        cmd=f"$BIN $D/automotive_susan_data/{i}.pgm output_large.corners.pgm -c",
        data=[f"automotive_susan_data/{i}.pgm"],
        outs=["output_large.corners.pgm"],
        linkopts=["-lm"],
    )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/tiff2bw",
    #     cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
    #     data=[f"consumer_tiff_data/{i}.tif"],
    #     outs=["output.tif"],
    #     linkopts=["-lm"],
    # )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/tiff2rgba",
    #     cmd=f"$BIN $D/consumer_tiff_data/{i}.tif output.tif",
    #     data=[f"consumer_tiff_data/{i}.tif"],
    #     outs=["output.tif"],
    #     linkopts=["-lm"],
    # )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/tiffdither",
    #     cmd=f"$BIN $D/consumer_tiff_data/{i}.bw.tif out.tif",
    #     data=[f"consumer_tiff_data/{i}.bw.tif"],
    #     outs=["out.tif"],
    #     linkopts=["-lm"],
    # )

    # TODO(cummins): Investigate.
    # validator(
    #     benchmark="benchmark://cBench-v0/tiffmedian",
    #     cmd=f"$BIN $D/consumer_tiff_data/{i}.nocomp.tif output.tif",
    #     data=[f"consumer_tiff_data/{i}.nocomp.tif"],
    #     outs=["output.tif"],
    #     linkopts=["-lm"],
    # )

    # NOTE(cummins): On macOS the following benchmarks abort with an illegal
    # hardware instruction error.
    if sys.platform != "darwin":
        # TODO(cummins): Investigate.
        # validator(
        #     benchmark="benchmark://cBench-v0/lame",
        #     cmd=f"$BIN $D/consumer_data/{i}.wav output.mp3",
        #     data=[f"consumer_data/{i}.wav"],
        #     outs=["output.mp3"],
        #     compare_output=False,
        #     linkopts=["-lm"],
        # )

        validator(
            benchmark="benchmark://cBench-v0/ghostscript",
            cmd=f"$BIN -sDEVICE=ppm -dNOPAUSE -dQUIET -sOutputFile=output.ppm -- {i}.ps",
            data=[f"office_data/{i}.ps"],
            outs=["output.ppm"],
            linkopts=["-lm", "-lz"],
            pre_execution_callback=setup_ghostscript_library_files,
        )
