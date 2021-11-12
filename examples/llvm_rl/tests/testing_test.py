# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from llvm_rl.model.testing import Testing
from omegaconf import OmegaConf

import compiler_gym


def test_testing_config():
    cfg = Testing(
        **OmegaConf.create(
            """\
timeout_hours: 12
runs_per_benchmark: 6
benchmarks:
    - dataset: benchmark://cbench-v1
      max_benchmarks: 5
"""
        )
    )
    assert cfg.timeout_hours == 12
    with compiler_gym.make("llvm-v0") as env:
        assert len(list(cfg.benchmark_uris_iterator(env))) == 5 * 6
