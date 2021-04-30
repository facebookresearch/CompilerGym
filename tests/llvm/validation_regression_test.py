# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for LlvmEnv.validate()."""
from io import StringIO

import pytest

from compiler_gym import CompilerEnvStateReader
from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


# The maximum number of times to call validate() on a state to check for an
# error.
VALIDATION_FLAKINESS = 3

# A list of CSV states that should pass validation, to be used as regression
# tests.
REGRESSION_TEST_STATES = """\
benchmark://cbench-v1/rijndael,,,opt -gvn -loop-unroll -instcombine -gvn -loop-unroll -instcombine input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -gvn -loop-unroll -mem2reg -loop-rotate -gvn -loop-unroll -mem2reg -loop-rotate input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -gvn-hoist input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -jump-threading -sink -partial-inliner -mem2reg -inline -jump-threading -sink -partial-inliner -mem2reg -inline input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -mem2reg -indvars -loop-unroll -simplifycfg -mem2reg -indvars -loop-unroll -simplifycfg input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -mem2reg -instcombine -early-cse-memssa -loop-unroll input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -reg2mem -licm -reg2mem -licm -reg2mem -licm input.bc -o output.bc
benchmark://cbench-v1/rijndael,,,opt -sroa -simplifycfg -partial-inliner input.bc -o output.bc
"""
REGRESSION_TEST_STATES = list(CompilerEnvStateReader(StringIO(REGRESSION_TEST_STATES)))
REGRESSION_TEST_STATE_NAMES = [
    f"{s.benchmark},{s.commandline}" for s in REGRESSION_TEST_STATES
]

# A list of CSV states that are known to fail validation.
KNOWN_BAD_STATES = """\
benchmark://cbench-v1/susan,0.40581008446378297,6.591785192489624,opt -mem2reg -reg2mem -gvn -reg2mem -gvn -newgvn input.bc -o output.bc
"""
KNOWN_BAD_STATES = list(CompilerEnvStateReader(StringIO(KNOWN_BAD_STATES)))
KNOWN_BAD_STATE_NAMES = [f"{s.benchmark},{s.commandline}" for s in KNOWN_BAD_STATES]
#
# NOTE(github.com/facebookresearch/CompilerGym/issues/103): The following
# regresison tests are deprecated after -structurizecfg was deactivated:
#
# benchmark://cbench-v1/tiff2bw,,,opt -structurizecfg input.bc -o output.bc
# benchmark://cbench-v1/tiff2rgba,,,opt -structurizecfg input.bc -o output.bc
# benchmark://cbench-v1/tiffdither,,,opt -structurizecfg input.bc -o output.bc
# benchmark://cbench-v1/tiffmedian,,,opt -structurizecfg input.bc -o output.bc
# benchmark://cBench-v0/susan,-0.5352209944751382,1.849454402923584,opt -structurizecfg -loop-extract -mergereturn -structurizecfg -loop-extract -mergereturn input.bc -o output.bc
# benchmark://cBench-v0/susan,0.9802486187845304,1.7552905082702637,opt -mem2reg -simplifycfg -lcssa -break-crit-edges -newgvn -mem2reg -simplifycfg -lcssa -break-crit-edges -newgvn input.bc -o output.bc


@skip_on_ci
@pytest.mark.parametrize("state", KNOWN_BAD_STATES, ids=KNOWN_BAD_STATE_NAMES)
def test_validate_known_bad_trajectory(env: LlvmEnv, state):
    env.apply(state)
    for _ in range(VALIDATION_FLAKINESS):
        result = env.validate()
        if result.okay():
            pytest.fail("Validation succeeded on state where it should have failed")


@skip_on_ci
@pytest.mark.parametrize(
    "state", REGRESSION_TEST_STATES, ids=REGRESSION_TEST_STATE_NAMES
)
def test_validate_known_good_trajectory(env: LlvmEnv, state):
    env.apply(state)
    for _ in range(VALIDATION_FLAKINESS):
        result = env.validate()
        if not result.okay():
            pytest.fail(f"Validation failed: {result}\n{result.dict()}")


if __name__ == "__main__":
    main()
