# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import sys
import tempfile
from pathlib import Path
from typing import Set

from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm import datasets
from compiler_gym.envs.llvm.datasets import get_llvm_benchmark_validation_callback
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

# The set of cBench benchmarks which do not have support for semantics
# validation.
CBENCH_VALIDATION_EXCLUDE_LIST: Set[str] = {
    "benchmark://cBench-v0/bzip2",
    "benchmark://cBench-v0/gsm",
    "benchmark://cBench-v0/ispell",
    "benchmark://cBench-v0/jpeg-c",
    "benchmark://cBench-v0/jpeg-d",
    "benchmark://cBench-v0/lame",
    "benchmark://cBench-v0/stringsearch2",
    "benchmark://cBench-v0/tiff2bw",
    "benchmark://cBench-v0/tiff2rgba",
    "benchmark://cBench-v0/tiffdither",
    "benchmark://cBench-v0/tiffmedian",
}
if sys.platform == "darwin":
    CBENCH_VALIDATION_EXCLUDE_LIST.add("benchmark://cBench-v0/ghostscript")


def test_no_validation_callback_for_custom_benchmark(env: LlvmEnv):
    """Test that a custom benchmark has no validation callback."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "example.c"
        with open(p, "w") as f:
            print("int main() {return 0;}", file=f)
        benchmark = env.make_benchmark(p)

    env.benchmark = benchmark
    env.reset()

    validation_cb = get_llvm_benchmark_validation_callback(env)

    assert validation_cb is None


def test_validate_cBench_unoptimized(env: LlvmEnv, benchmark_name: str):
    """Run the validation routine on unoptimized version of all cBench benchmarks."""
    env.reset(benchmark=benchmark_name)
    cb = datasets.get_llvm_benchmark_validation_callback(env)

    if benchmark_name in CBENCH_VALIDATION_EXCLUDE_LIST:
        assert cb is None
    else:
        assert cb
        assert cb(env) is None


if __name__ == "__main__":
    main()
