# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.commandline()."""
import os
import subprocess
from pathlib import Path

import pytest

from compiler_gym.envs import LlvmEnv
from compiler_gym.util.commands import Popen
from tests.pytest_plugins.random_util import apply_random_trajectory
from tests.test_main import main

pytest_plugins = [
    "tests.pytest_plugins.llvm",
    "tests.pytest_plugins.common",
]

# The uniform range for trajectory lengths.
RANDOM_TRAJECTORY_LENGTH_RANGE = (1, 50)


@pytest.mark.timeout(600)
def test_fuzz(env: LlvmEnv, tmpwd: Path, llvm_opt: Path, llvm_diff: Path):
    """This test produces a random trajectory and then uses the commandline()
    generated with opt to check that the states are equivalent.
    """
    del tmpwd

    env.reset()
    env.write_ir("input.ll")
    assert Path("input.ll").is_file()

    # In case of a failure, create a regression test by copying the body of this
    # function and replacing the below line with the commandline printed below.
    apply_random_trajectory(
        env, random_trajectory_length_range=RANDOM_TRAJECTORY_LENGTH_RANGE, timeout=30
    )
    commandline = env.commandline(textformat=True)
    print(env.state)  # For debugging in case of failure.

    # Write the post-trajectory state to file.
    env.write_ir("env.ll")
    assert Path("env.ll").is_file()

    # Run the environment commandline using LLVM opt.
    subprocess.check_call(
        commandline, env={"PATH": str(llvm_opt.parent)}, shell=True, timeout=60
    )
    assert Path("output.ll").is_file()
    os.rename("output.ll", "opt.ll")

    with Popen(
        [llvm_diff, "opt.ll", "env.ll"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ) as diff:
        stdout, stderr = diff.communicate(timeout=300)
        if diff.returncode:
            pytest.fail(
                f"Opt produced different output to CompilerGym "
                f"(returncode: {diff.returncode}):\n{stdout}\n{stderr}"
            )


if __name__ == "__main__":
    main()
