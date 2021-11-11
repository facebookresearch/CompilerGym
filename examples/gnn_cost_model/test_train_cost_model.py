"""Tests for //compiler_gym/bin:train.py"""


import sys

sys.path.append("./examples/gnn_cost_model")

from absl import flags
from train_cost_model import main

from compiler_gym.util.capture_output import capture_output

FLAGS = flags.FLAGS


def test_run_train_smoke_test():
    flags = [
        "argv0",
        "--dataset_size=64",
        "--num_epoch=2",
    ]
    sys.argv = flags
    FLAGS(flags)
    with capture_output() as out:
        main(["argv0"])

    assert "Epoch num 0 training" in out.stdout
