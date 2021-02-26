# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fuzz test for LlvmEnv.commandline()."""
import subprocess
from difflib import unified_diff
from pathlib import Path

import pytest

from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.random_util import apply_random_trajectory
from tests.test_main import main

pytest_plugins = [
    "tests.pytest_plugins.llvm",
    "tests.pytest_plugins.common",
]

# The uniform range for trajectory lengths.
RANDOM_TRAJECTORY_LENGTH_RANGE = (1, 50)


def test_fuzz(env: LlvmEnv, tmpwd: Path, llvm_opt: Path, llvm_diff: Path):
    """This test produces a random trajectory and then uses the commandline()
    generated with opt to check that the states are equivalent.
    """
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

    with open("output.ll") as f1, open("env.ll") as f2:
        # Diff the IR files but exclude the first line which is the module name.
        diff = list(unified_diff(f1.readlines()[1:], f2.readlines()[1:]))

        if diff and len(diff) < 25:
            diff = "\n".join(diff)
            pytest.fail(f"Opt produced different output to CompilerGym:\n{diff}")
        elif diff:
            # If it's a big diff then we will require the user to reproduce it
            # themselves using the environment state we printed earlier.
            pytest.fail(
                f"Opt produced different output to CompilerGym ({len(diff)}-line diff)"
            )


if __name__ == "__main__":
    main()
