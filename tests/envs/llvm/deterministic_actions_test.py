# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for action space determinism."""
import hashlib

import pytest

from compiler_gym.envs import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


def test_deterministic_action(env: LlvmEnv, benchmark_name: str, action_name: str):
    env.observation_space = "Ir"

    checksums = set()
    for i in range(1, 21):
        ir = env.reset(benchmark=benchmark_name)
        sha1 = hashlib.sha1()
        sha1.update(ir.encode("utf-8"))
        checksum_before = sha1.hexdigest()

        ir, _, done, _ = env.step(env.action_space.names.index(action_name))
        assert not done
        sha1 = hashlib.sha1()
        sha1.update(ir.encode("utf-8"))
        checksums.add(sha1.hexdigest())

        if len(checksums) != 1:
            pytest.fail(
                f"Repeating the {action_name} action {i} times on {benchmark_name} "
                "produced different states"
            )

        # An action which has no effect is not likely to be nondeterministic.
        if list(checksums)[0] == checksum_before:
            break


if __name__ == "__main__":
    main()
