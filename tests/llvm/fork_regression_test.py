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
        ForkRegressionTest(
            benchmark="benchmark://cbench-v1/qsort",
            pre_fork="-loop-versioning -barrier -deadargelim -loop-guard-widening -elim-avail-extern -elim-avail-extern -lowerinvoke -strip-debug-declare -name-anon-globals -strip-nondebug -rewrite-statepoints-for-gc -redundant-dbg-inst-elim -correlated-propagation -adce -deadargelim -globalopt -div-rem-pairs -elim-avail-extern -nary-reassociate -lowerinvoke -canonicalize-aliases -sancov -inferattrs -loop-reroll -loop-deletion -dse -name-anon-globals -inferattrs -callsite-splitting -alignment-from-assumptions -inferattrs -early-cse -functionattrs -jump-threading -loop-instsimplify -reassociate -flattencfg -memcpyopt -canonicalize-aliases -post-inline-ee-instrument -tailcallelim -lower-matrix-intrinsics -argpromotion -early-cse -inline -lower-constant-intrinsics -die -prune-eh -mergeicmps -pgo-memop-opt -simplifycfg -called-value-propagation -simplifycfg -loop-data-prefetch -loop-reroll -simplifycfg -div-rem-pairs -sccp -slp-vectorizer -ipsccp -separate-const-offset-from-gep -loop-vectorize -sroa -loop-simplifycfg -loop-load-elim -reassociate -loop-distribute -canonicalize-aliases -strip-dead-prototypes -attributor -callsite-splitting -mergereturn -mldst-motion -strip -rpo-functionattrs -dse -loop-idiom -guard-widening -hotcoldsplit -lcssa -loweratomic -prune-eh -newgvn -tailcallelim -prune-eh -rpo-functionattrs -slp-vectorizer -inferattrs -always-inline -float2int -lower-guard-intrinsic -lower-constant-intrinsics -simple-loop-unswitch -loop-versioning -instcombine -loweratomic -add-discriminators -inline -loop-deletion -slp-vectorizer -flattencfg -loop-unroll-and-jam -dse -dse -lower-widenable-condition -loop-rotate -hotcoldsplit -early-cse -mem2reg -tailcallelim -slp-vectorizer -cross-dso-cfi -coro-split -dce -memcpyopt -alignment-from-assumptions -coro-early -sink -loop-versioning -attributor -partially-inline-libcalls -coro-early -instcombine -lower-expect -constprop -loop-unswitch -loop-versioning -rpo-functionattrs -nary-reassociate -gvn -lower-guard-intrinsic -loop-unroll-and-jam -attributor -loop-idiom -lcssa -loop-load-elim -speculative-execution -float2int -mergefunc -lowerswitch -elim-avail-extern -coro-cleanup -scalarizer -redundant-dbg-inst-elim -load-store-vectorizer -instnamer -mem2reg -lower-matrix-intrinsics -insert-gcov-profiling -hotcoldsplit -loop-instsimplify -lowerinvoke -coro-early -coro-early -slp-vectorizer -coro-split -deadargelim -break-crit-edges -pgo-memop-opt -gvn-hoist -loop-instsimplify -loop-data-prefetch -gvn -newgvn -ee-instrument -strip-nondebug -alignment-from-assumptions -inline -mergefunc -adce -coro-cleanup -prune-eh",
            post_fork="-lcssa ",
        ),
        ForkRegressionTest(
            benchmark="benchmark://poj104-v1/101/859",
            pre_fork="-bdce -ipconstprop -forceattrs -reg2mem -deadargelim -adce -lower-expect -instsimplify -sink -loop-simplifycfg -inline -loop-unroll-and-jam -sroa -loop-predication -bdce -loop-fusion -sink -float2int -alignment-from-assumptions -licm -strip-debug-declare -dce -sroa -aggressive-instcombine -loop-distribute -rewrite-statepoints-for-gc -slsr -bdce -prune-eh -forceattrs -constprop -name-anon-globals -canonicalize-aliases -deadargelim -loop-simplifycfg -partially-inline-libcalls -die -libcalls-shrinkwrap -called-value-propagation -coro-split -loop-idiom -loop-idiom -mergeicmps -elim-avail-extern -jump-threading -constmerge -canonicalize-aliases -loop-simplifycfg -licm",
            post_fork="-loop-rotate ",
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

    _, _, done, info = env.multistep(pre_fork)
    assert not done, info

    with env.fork() as fkd:
        assert env.state == fkd.state  # Sanity check

        env.multistep(post_fork)
        fkd.multistep(post_fork)
        # Verify that the environment states no longer line up.
        assert env.state != fkd.state


if __name__ == "__main__":
    main()
