# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
from llvm_rl.model.training import Benchmarks
from omegaconf import OmegaConf
from pydantic import ValidationError

import compiler_gym
from compiler_gym.datasets import Benchmark


def test_benchmarks_missing_dataset_and_uris():
    with pytest.raises(ValidationError):
        Benchmarks()


def test_benchmarks_uris_list():
    cfg = Benchmarks(uris=["benchmark://cbench-v1/qsort"])
    assert cfg.uris == ["benchmark://cbench-v1/qsort"]

    with compiler_gym.make("llvm-v0") as env:
        assert list(cfg.benchmarks_iterator(env)) == ["benchmark://cbench-v1/qsort"]
        assert isinstance(list(cfg.benchmarks_iterator(env))[0], Benchmark)
        assert list(cfg.benchmark_uris_iterator(env)) == ["benchmark://cbench-v1/qsort"]


def test_validation_benchmarks_uris_list_yaml():
    cfg = Benchmarks(
        **OmegaConf.create(
            """\
uris:
  - benchmark://cbench-v1/qsort
"""
        )
    )
    assert len(cfg.uris) == 1
