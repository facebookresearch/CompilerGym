# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import explore


def test_run_explore_smoke_test(capsys):
    explore.main(
        [
            "explore",
            "--env=llvm-ic-v0",
            "--benchmark=cbench-v1/dijkstra",
            "--episode_length=2",
            "--explore_actions=-newgvn,-instcombine,-mem2reg",
            "--nproc=2",
        ]
    )
    out, _ = capsys.readouterr()
    assert "depth 2 of 2" in out
