# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""End-to-end test of //compiler_gym/bin:action_sensitivity_analysis."""

import tempfile
from pathlib import Path

from examples.sensitivity_analysis.action_sensitivity_analysis import (
    run_action_sensitivity_analysis,
)
from examples.sensitivity_analysis.sensitivity_analysis_eval import (
    run_sensitivity_analysis_eval,
)
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main


def test_run_action_sensitivity_analysis():
    actions = [0, 1]
    env = "llvm-v0"
    reward = "IrInstructionCountO3"
    benchmark = "cbench-v1/crc32"

    set_command_line_flags(["argv0", f"--env={env}", f"--benchmark={benchmark}"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_action_sensitivity_analysis(
            actions=actions,
            rewards_path=tmp / "rewards.txt",
            runtimes_path=tmp / "runtimes.txt",
            reward_space=reward,
            num_trials=2,
            max_warmup_steps=5,
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
