# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from llvm_rl.model.training import Training
from omegaconf import OmegaConf


def test_parse_yaml():
    cfg = Training(
        **OmegaConf.create(
            """\
timeout_hours: 10
episodes: 1000
benchmarks:
    - uris:
        - benchmark://cbench-v1/qsort
    - dataset: benchmark://cbench-v1
      max_benchmarks: 2
validation:
    benchmarks:
        - uris:
            - benchmark://cbench-v1/qsort
"""
        )
    )
    assert cfg.timeout_hours == 10
