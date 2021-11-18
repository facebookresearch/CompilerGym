# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from llvm_rl.model.validation import Validation
from omegaconf import OmegaConf

import compiler_gym
from compiler_gym.datasets import Benchmark


def test_validation_benchmarks_uris_list():
    cfg = Validation(
        **OmegaConf.create(
            """\
benchmarks:
    - uris:
        - benchmark://cbench-v1/qsort
    - dataset: benchmark://cbench-v1
      max_benchmarks: 2
"""
        )
    )

    with compiler_gym.make("llvm-v0") as env:
        assert list(cfg.benchmarks_iterator(env)) == [
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/adpcm",
            "benchmark://cbench-v1/bitcount",
        ]
        bm = list(cfg.benchmarks_iterator(env))[0]
        print(type(bm).__name__)
        assert isinstance(bm, Benchmark)
        assert list(cfg.benchmark_uris_iterator(env)) == [
            "benchmark://cbench-v1/qsort",
            "benchmark://cbench-v1/adpcm",
            "benchmark://cbench-v1/bitcount",
        ]
