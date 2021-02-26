# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for action space determinism."""
import hashlib
import random

import pytest

from compiler_gym.envs import LlvmEnv
from tests.pytest_plugins.llvm import BENCHMARK_NAMES
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


ACTION_REPTITION_COUNT = 20


def sha1(string: str):
    sha1 = hashlib.sha1()
    sha1.update(string.encode("utf-8"))
    return sha1.hexdigest()


def test_fuzz(env: LlvmEnv):
    """Run an action multiple times from the same starting state and check that
    the generated LLVM-IR is the same.

    Caveats of this test:

        * The initial state is an unoptimized benchmark. If a pass depends
          on other passes to take effect it will not be tested.

        * Non-determinism is tested by running the action 20 times. Extremely
          unlikely non-determinism may not be detected.
    """
    action = env.action_space.sample()
    action_name = env.action_space.names[action]
    benchmark = random.choice(BENCHMARK_NAMES)

    env.observation_space = "Ir"

    checksums = set()
    for i in range(1, ACTION_REPTITION_COUNT + 1):
        ir = env.reset(benchmark=benchmark)
        checksum_before = sha1(ir)

        ir, _, done, _ = env.step(action)
        assert not done
        checksums.add(sha1(ir))

        if len(checksums) != 1:
            pytest.fail(
                f"Repeating the {action_name} action {i} times on "
                f"{benchmark} produced different states"
            )

        # An action which has no effect is not likely to be nondeterministic.
        if list(checksums)[0] == checksum_before:
            break


if __name__ == "__main__":
    main()
