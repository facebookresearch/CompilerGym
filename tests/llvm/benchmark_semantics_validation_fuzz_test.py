# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import sys
from typing import Set

from compiler_gym.envs import LlvmEnv
from compiler_gym.envs.llvm.datasets import get_llvm_benchmark_validation_callback
from tests.test_main import main

pytest_plugins = ["tests.llvm.fixtures"]

# The set of cBench benchmarks which do not have support for semantics
# validation.
CBENCH_VALIDATION_EXCLUDE_LIST: Set[str] = {
    "benchmark://cBench-v0/ispell",
    "benchmark://cBench-v0/stringsearch2",
}
if sys.platform == "darwin":
    CBENCH_VALIDATION_EXCLUDE_LIST.add("benchmark://cBench-v0/lame")
    CBENCH_VALIDATION_EXCLUDE_LIST.add("benchmark://cBench-v0/ghostscript")


def test_validate_cBench_unoptimized(env: LlvmEnv, benchmark_name: str):
    """Run the validation routine on unoptimized version of all cBench benchmarks."""
    env.reset(benchmark=benchmark_name)
    cb = get_llvm_benchmark_validation_callback(env)

    if benchmark_name in CBENCH_VALIDATION_EXCLUDE_LIST:
        assert cb is None
    else:
        assert cb
        assert cb(env) is None


if __name__ == "__main__":
    main()
