# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for LlvmEnv.fork() identified by hand or through fuzzing."""

from typing import List, NamedTuple

import pytest
from flaky import flaky

import compiler_gym
from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


class ForkRegressionTest(NamedTuple):
    benchmark: str
    pre_fork: str
    post_fork: str
    reward_space: str = "IrInstructionCount"


# A list of testcases where we have identified the parent and child environment
# states differing after forking and running identical actions on both.
#
# NOTE(cummins): To submit a new testcase, run the
# "minimize_fork_regression_testcase()" function below to produce a minimal
# reproducible example and add it to this list.
MINIMIZED_FORK_REGRESSION_TESTS: List[ForkRegressionTest] = [
    ForkRegressionTest(
        benchmark="benchmark://cbench-v1/tiff2bw",
        pre_fork="-globalopt",
        post_fork="-gvn-hoist",
        reward_space="IrInstructionCount",
    ),
    ForkRegressionTest(
        benchmark="benchmark://cbench-v1/bzip2",
        pre_fork="-mem2reg",
        post_fork="-loop-guard-widening",
        reward_space="IrInstructionCount",
    ),
    ForkRegressionTest(
        benchmark="benchmark://cbench-v1/jpeg-d",
        pre_fork="-sroa",
        post_fork="-loop-rotate",
        reward_space="IrInstructionCount",
    ),
    ForkRegressionTest(
        benchmark="benchmark://cbench-v1/qsort",
        pre_fork="-simplifycfg -newgvn -instcombine -break-crit-edges -gvn -inline",
        post_fork="-lcssa",
        reward_space="IrInstructionCount",
    ),
    ForkRegressionTest(
        benchmark="benchmark://poj104-v1/101/859",
        pre_fork="-licm",
        post_fork="-loop-rotate",
        reward_space="IrInstructionCount",
    ),
]


@flaky
@pytest.mark.parametrize("test", MINIMIZED_FORK_REGRESSION_TESTS)
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

    _, _, done, info = env.multistep(pre_fork)
    assert not done, info

    with env.fork() as fkd:
        assert env.state == fkd.state  # Sanity check

        env.multistep(post_fork)
        fkd.multistep(post_fork)
        # Verify that the environment states no longer line up.
        assert env.state != fkd.state


# Utility function for generating test cases. Copy this code into a standalone
# script and call the function on your test case. It will print a minimized
# version of it.


def minimize_fork_regression_testcase(test: ForkRegressionTest):
    def _check_hypothesis(
        benchmark: str, pre_fork: List[int], post_fork: List[int]
    ) -> bool:
        with compiler_gym.make("llvm-v0", reward_space="IrInstructionCount") as env:
            env.reset(benchmark)
            _, _, done, info = env.multistep(pre_fork)
            assert not done, info  # Sanity check
            with env.fork() as fkd:
                assert env.state == fkd.state  # Sanity check
                env.multistep(post_fork)
                fkd.multistep(post_fork)
                return env.state != fkd.state

    with compiler_gym.make("llvm-v0", reward_space=test.reward_space) as env:
        pre_fork = [env.action_space[f] for f in test.pre_fork.split()]
        post_fork = [env.action_space[f] for f in test.post_fork.split()]

        pre_fork_mask = [True] * len(pre_fork)
        post_fork_mask = [True] * len(post_fork)

        print("Minimizing the pre-fork actions list")
        for i in range(len(pre_fork)):
            pre_fork_mask[i] = False
            masked_pre_fork = [p for p, m in zip(pre_fork, pre_fork_mask) if m]
            if _check_hypothesis(test.benchmark, masked_pre_fork, post_fork):
                print(
                    f"Removed pre-fork action {env.action_space.names[pre_fork[i]]}, {sum(pre_fork_mask)} remaining"
                )
            else:
                pre_fork_mask[i] = True
        pre_fork = [p for p, m in zip(pre_fork, pre_fork_mask) if m]

        print("Minimizing the post-fork actions list")
        for i in range(len(post_fork)):
            post_fork_mask[i] = False
            masked_post_fork = [p for p, m in zip(post_fork, post_fork_mask) if m]
            if _check_hypothesis(test.benchmark, pre_fork, masked_post_fork):
                print(
                    f"Removed post-fork action {env.action_space.names[post_fork[i]]}, {sum(post_fork_mask)} remaining"
                )
            else:
                pre_fork_mask[i] = True
        post_fork = [p for p, m in zip(post_fork, post_fork_mask) if m]

        pre_fork = " ".join(env.action_space.names[p] for p in pre_fork)
        post_fork = " ".join(env.action_space.names[p] for p in post_fork)

    return ForkRegressionTest(
        benchmark=test.benchmark,
        pre_fork=pre_fork,
        post_fork=post_fork,
        reward_space=test.reward_space,
    )


if __name__ == "__main__":
    main()
