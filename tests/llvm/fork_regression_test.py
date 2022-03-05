# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for LlvmEnv.fork() identified by hand or through fuzzing."""

from typing import NamedTuple

import pytest
from flaky import flaky

from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


class ForkRegressionTest(NamedTuple):
    benchmark: str
    pre_fork: str
    post_fork: str
    reward_space: str = "IrInstructionCount"


@flaky
@pytest.mark.parametrize(
    "test",
    [
        ForkRegressionTest(
            benchmark="benchmark://cbench-v1/tiff2bw",
            pre_fork="-loop-unswitch -name-anon-globals -attributor -correlated-propagation -loop-unroll-and-jam -reg2mem -break-crit-edges -globalopt -inline",
            post_fork="-cross-dso-cfi -gvn-hoist",
        ),
        ForkRegressionTest(
            benchmark="benchmark://cbench-v1/bzip2",
            pre_fork="-loop-deletion -argpromotion -callsite-splitting -mergeicmps -deadargelim -instsimplify -mem2reg -instcombine -rewrite-statepoints-for-gc -insert-gcov-profiling",
            post_fork="-partially-inline-libcalls -loop-guard-widening",
        ),
        ForkRegressionTest(
            benchmark="benchmark://cbench-v1/jpeg-d",
            pre_fork="-bdce -loop-guard-widening -loop-reduce -globaldce -sroa -partially-inline-libcalls -loop-deletion -forceattrs -flattencfg -simple-loop-unswitch",
            post_fork="-mergefunc -dse -load-store-vectorizer -sroa -mldst-motion -hotcoldsplit -loop-versioning-licm -loop-rotate",
        ),
    ],
)
def test_fork_regression_test(env: LlvmEnv, test: ForkRegressionTest):
    """Run the fork regression test:

    1. Initialize an environment.
    2. Apply a "pre_fork" sequence of actions.
    3. Create a fork of the environment.
    4. Apply a "post_fork" sequence of actions in both the fork and parent.
    5. Verify that the environment states have gone out of sync.
    """
    env.reward_space = test.reward_space
    env.reset(test.benchmark)
    pre_fork = [env.action_space[f] for f in test.pre_fork.split()]
    post_fork = [env.action_space[f] for f in test.post_fork.split()]

    for action in pre_fork:
        _, _, done, info = env.step(action)
        assert not done, info

    with env.fork() as fkd:
        assert env.state == fkd.state  # Sanity check

        for action in post_fork:
            env.step(action)
            fkd.step(action)
        # Verify that the environment states no longer line up.
        assert env.state != fkd.state


if __name__ == "__main__":
    main()
