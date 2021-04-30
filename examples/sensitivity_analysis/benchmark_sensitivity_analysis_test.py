# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end test of //compiler_gym/bin:benchmark_sensitivity_analysis."""

import tempfile
from pathlib import Path

from examples.sensitivity_analysis.benchmark_sensitivity_analysis import (
    run_benchmark_sensitivity_analysis,
)
from examples.sensitivity_analysis.sensitivity_analysis_eval import (
    run_sensitivity_analysis_eval,
)
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main


def test_run_benchmark_sensitivity_analysis():
    env = "llvm-v0"
    reward = "IrInstructionCountO3"
    benchmarks = ["cbench-v1/crc32"]

    set_command_line_flags(["argv0", f"--env={env}"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_benchmark_sensitivity_analysis(
            benchmarks=benchmarks,
            rewards_path=tmp / "rewards.txt",
            runtimes_path=tmp / "runtimes.txt",
            reward=reward,
            num_trials=2,
            min_steps=3,
            max_steps=5,
            nproc=1,
        )

        assert (tmp / "rewards.txt").is_file()
        assert (tmp / "runtimes.txt").is_file()

        run_sensitivity_analysis_eval(
            rewards_path=tmp / "rewards.txt",
            runtimes_path=tmp / "runtimes.txt",
        )


if __name__ == "__main__":
    main()
