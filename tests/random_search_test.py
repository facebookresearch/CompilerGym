# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:random_search."""
import tempfile
from pathlib import Path

import gym

from compiler_gym.random_replay import replay_actions_from_logs
from compiler_gym.random_search import random_search
from tests.pytest_plugins.common import set_command_line_flags
from tests.test_main import main


def make_env():
    env = gym.make("llvm-autophase-ic-v0")
    env.benchmark = "cbench-v1/dijkstra"
    return env


def test_random_search_smoke_test():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        set_command_line_flags(["argv0"])
        random_search(
            make_env=make_env,
            outdir=outdir,
            patience=50,
            total_runtime=3,
            nproc=1,
            skip_done=False,
        )

        assert (outdir / "random_search.json").is_file()
        assert (outdir / "random_search_progress.csv").is_file()
        assert (outdir / "random_search_best_actions.txt").is_file()
        assert (outdir / "optimized.bc").is_file()

        env = make_env()
        try:
            replay_actions_from_logs(env, Path(outdir))
            assert (outdir / "random_search_best_actions_progress.csv").is_file()
            assert (outdir / "random_search_best_actions_commandline.txt").is_file()
        finally:
            env.close()


if __name__ == "__main__":
    main()
