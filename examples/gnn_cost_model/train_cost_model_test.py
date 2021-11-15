# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for examples/gnn_cost_model/train_cost_model_test.py"""
import sys

from absl import flags
from train_cost_model import main

from compiler_gym.util.capture_output import capture_output

FLAGS = flags.FLAGS


def test_run_train_smoke_test():
    flags = [
        "argv0",
        "--dataset_size=64",
        "--batch_size=4",
        "--num_epoch=2",
    ]
    sys.argv = flags
    FLAGS(flags)
    with capture_output() as out:
        main(["argv0"])

    assert "Epoch num 0 training" in out.stdout
