# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for action space determinism."""
import hashlib

import pytest

from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]

ACTION_REPTITION_COUNT = 50


@pytest.mark.skip(reason="github.com/facebookresearch/CompilerGym/issues/46")
@pytest.mark.parametrize(
    "benchmark_name",
    [
        "benchmark://cBench-v0/adpcm",
        "benchmark://cBench-v0/bitcount",
        "benchmark://cBench-v0/blowfish",
        "benchmark://cBench-v0/bzip2",
        "benchmark://cBench-v0/ghostscript",
        "benchmark://cBench-v0/gsm",
        "benchmark://cBench-v0/ispell",
        "benchmark://cBench-v0/jpeg-c",
        "benchmark://cBench-v0/jpeg-d",
        "benchmark://cBench-v0/patricia",
        "benchmark://cBench-v0/rijndael",
        "benchmark://cBench-v0/stringsearch",
        "benchmark://cBench-v0/stringsearch2",
        "benchmark://cBench-v0/susan",
        "benchmark://cBench-v0/tiff2bw",
        "benchmark://cBench-v0/tiff2rgba",
        "benchmark://cBench-v0/tiffdither",
        "benchmark://cBench-v0/tiffmedian",
    ],
)
def test_gvn_sink_non_determinism(env: LlvmEnv, benchmark_name: str):
    """Regression test for -gvn-sink non-determinism.
    See: https://github.com/facebookresearch/CompilerGym/issues/46
    """
    env.observation_space = "Ir"

    checksums = set()
    for i in range(1, ACTION_REPTITION_COUNT + 1):
        env.reset(benchmark=benchmark_name)
        ir, _, done, _ = env.step(env.action_space.names.index("-gvn-sink"))
        assert not done
        sha1 = hashlib.sha1()
        sha1.update(ir.encode("utf-8"))
        checksums.add(sha1.hexdigest())

        if len(checksums) != 1:
            pytest.fail(
                f"Repeating the -gvn-sink action {i} times on {benchmark_name} "
                "produced different states"
            )


if __name__ == "__main__":
    main()
