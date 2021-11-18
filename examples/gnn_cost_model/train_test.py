# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for examples/gnn_cost_model/train_cost_model_test.py"""
import sys

import pytest
from absl import flags

from compiler_gym.util.capture_output import capture_output

from .train import main

FLAGS = flags.FLAGS


@pytest.mark.skip(reason="Need to create a small test set")
def test_run_train_smoke_test():
    flags = [
        "argv0",
        "--dataset_size=64",
        "--batch_size=4",
        "--num_epoch=2",
        "--device=cpu",
    ]
    sys.argv = flags
    FLAGS(flags)
    with capture_output() as out:
        main(["argv0"])

    assert "Epoch num 0 training" in out.stdout
