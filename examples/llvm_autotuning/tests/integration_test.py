# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import sys
from pathlib import Path


def test_llvm_autotuner_integration_test(tmp_path: Path):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "llvm_autotuning.tune",
            "-m",
            "experiment=my-exp",
            f"outputs={tmp_path}/llvm_autotuning",
            "executor.cpus=1",
            "num_replicas=1",
            "autotuner=nevergrad",
            "autotuner.optimization_target=codesize",
            "autotuner.search_time_seconds=3",
            "autotuner.algorithm_config.episode_length=5",
            "benchmarks=single_benchmark_for_testing",
        ]
    )
    assert (Path(tmp_path) / "llvm_autotuning/my-exp").is_dir()
