# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

import gcc_search
from absl.flags import FLAGS

from compiler_gym.service import EnvironmentNotSupported


def test_gcc_search_smoke_test(capsys):
    flags = [
        "argv0",
        "--seed=0",
        "--episode_len=2",
        "--episodes=10",
        "--log_interval=5",
        "--benchmark=cbench-v1/crc32",
    ]
    sys.argv = flags
    FLAGS.unparse_flags()
    FLAGS(flags)

    try:
        gcc_search.main(
            [
                "gcc_search",
                "--gcc_benchmark=benchmark://chstone-v0/aes",
                "--search=random",
                "--n=3",
            ]
        )
        out, _ = capsys.readouterr()
        assert "benchmark://chstone-v0/aes" in out
    except EnvironmentNotSupported:
        pass  # GCC environment might not be supported
