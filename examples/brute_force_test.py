# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:brute_force."""
import tempfile
from pathlib import Path

import gym

from examples.brute_force import run_brute_force
from tests.test_main import main


def test_run_brute_force_smoke_test():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        run_brute_force(
            make_env=lambda: gym.make("llvm-ic-v0", benchmark="cbench-v1/crc32"),
            action_names=["-sroa", "-mem2reg"],
            episode_length=2,
            outdir=outdir,
            nproc=1,
            chunksize=2,
        )

        assert (outdir / "meta.json").is_file()
        assert (outdir / "results.csv").is_file()


if __name__ == "__main__":
    main(extra_pytest_args=["-s"], debug_level=2)  # Don't capture stdout/stderr.
