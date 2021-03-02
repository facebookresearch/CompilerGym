# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for LlvmEnv.validate()."""
import pytest

from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.common import linux_only
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.mark.xfail(
    strict=True, reason="github.com/facebookresearch/CompilerGym/issues/103"
)
def test_validate_strucutrizecfg_stringsearch(env: LlvmEnv):
    # Regression test for a failure caused by -structurizecfg.
    env.reset("cBench-v0/stringsearch")
    env.step(env.action_space["-instcombine"])
    env.step(env.action_space["-jump-threading"])
    env.step(env.action_space["-loop-interchange"])
    env.step(env.action_space["-structurizecfg"])
    assert env.validate().okay()


@linux_only
@pytest.mark.xfail(
    strict=True, reason="github.com/facebookresearch/CompilerGym/issues/103"
)
def test_validate_strucutrizecfg_ghostscript(env: LlvmEnv):
    # Regression test for a failure caused by -structurizecfg.
    env.reset("cBench-v0/ghostscript")
    env.step(env.action_space["-structurizecfg"])
    assert env.validate().okay()


@pytest.mark.xfail(
    strict=True, reason="github.com/facebookresearch/CompilerGym/issues/105"
)
def test_validate_susan_a(env: LlvmEnv):
    env.reset("cBench-v0/susan")
    env.step(env.action_space["-structurizecfg"])
    env.step(env.action_space["-loop-extract"])
    env.step(env.action_space["-mergereturn"])
    env.step(env.action_space["-structurizecfg"])
    env.step(env.action_space["-loop-extract"])
    env.step(env.action_space["-mergereturn"])
    assert env.validate().okay()


@pytest.mark.xfail(
    strict=True, reason="github.com/facebookresearch/CompilerGym/issues/105"
)
def test_validate_susan_b(env: LlvmEnv):
    env.reset("cBench-v0/susan")
    env.step(env.action_space["-mem2reg"])
    env.step(env.action_space["-simplifycfg"])
    env.step(env.action_space["-lcssa"])
    env.step(env.action_space["-break-crit-edges"])
    env.step(env.action_space["-newgvn"])
    env.step(env.action_space["-mem2reg"])
    env.step(env.action_space["-simplifycfg"])
    env.step(env.action_space["-lcssa"])
    env.step(env.action_space["-break-crit-edges"])
    env.step(env.action_space["-newgvn"])
    assert env.validate().okay()


if __name__ == "__main__":
    main()
