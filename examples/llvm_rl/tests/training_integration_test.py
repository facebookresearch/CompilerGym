# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys
import warnings
from pathlib import Path

from llvm_rl.model.model import Model
from omegaconf import OmegaConf


def test_local_train(tmp_path: Path):
    model = Model(
        **OmegaConf.create(
            f"""\
experiment: tiger
working_directory: {tmp_path}/outputs
executor:
    type: local
    cpus: 2
environment:
    id: llvm-autophase-ic-v0
    max_episode_steps: 3
agent:
    type: PPOTrainer
    args:
        lr: 1.e-3
        model:
            fcnet_hiddens: [16]
            fcnet_activation: relu
        framework: torch
        rollout_fragment_length: 8
        train_batch_size: 8
        sgd_minibatch_size: 8
training:
    timeout_hours: 0.25
    episodes: 32
    benchmarks:
        - dataset: benchmark://cbench-v1
          max_benchmarks: 3
    validation:
        benchmarks:
            - dataset: benchmark://cbench-v1
              max_benchmarks: 3
testing:
    timeout_hours: 0.25
    benchmarks:
        - dataset: benchmark://cbench-v1
          max_benchmarks: 3
"""
        )
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    model.train()

    print("Outputs", list((tmp_path / "outputs").iterdir()), file=sys.stderr)
    assert (tmp_path / "outputs").is_dir()
    with open(tmp_path / "outputs" / "training-model.json") as f:
        assert json.load(f)

    assert (tmp_path / "outputs" / "train").is_dir()
    print("Outputs", list((tmp_path / "outputs" / "train").iterdir()), file=sys.stderr)

    # Check that a checkpoint was created.
    assert (
        tmp_path
        / "outputs"
        / "train"
        / "tiger-C0-R0"
        / "checkpoint_000001"
        / "checkpoint-1"
    ).is_file()

    # TODO(github.com/facebookresearch/CompilerGym/issues/487): Fix test on CI.
    if os.environ.get("CI", "") != "":
        return

    model.test()
    print(
        "Trail files",
        list((tmp_path / "outputs" / "train" / "tiger-C0-R0").iterdir()),
        file=sys.stderr,
        flush=True,
    )
    assert (tmp_path / "outputs" / "train" / "tiger-C0-R0" / "test-meta.json").is_file()
    assert (
        tmp_path / "outputs" / "train" / "tiger-C0-R0" / "test-results.json"
    ).is_file()
