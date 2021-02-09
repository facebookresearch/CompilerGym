# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for action space determinism."""
import hashlib

import pytest

from compiler_gym.envs import LlvmEnv
from tests.fixtures import skip_on_ci
from tests.test_main import main

pytest_plugins = ["tests.envs.llvm.fixtures"]


ACTION_REPTITION_COUNT = 20


def sha1(string: str):
    sha1 = hashlib.sha1()
    sha1.update(string.encode("utf-8"))
    return sha1.hexdigest()


@skip_on_ci
def test_deterministic_action(env: LlvmEnv, benchmark_name: str, action_name: str):
    """Run an action multiple times from the same starting state and check that
    the generated LLVM-IR is the same.

    Do this for every combination of benchmark and action. This generates many
    tests.

    Caveats of this test:

        * The initial states are all unoptimized benchmarks. If a pass depends
          on other passes to take effect it will not be tested.

        * Non-determinism is tested by running the action 20 times. Extremely
          unlikely non-determinism may not be detected.
    """
    env.observation_space = "Ir"

    checksums = set()
    for i in range(1, ACTION_REPTITION_COUNT + 1):
        ir = env.reset(benchmark=benchmark_name)
        checksum_before = sha1(ir)

        ir, _, done, _ = env.step(env.action_space.names.index(action_name))
        assert not done
        checksums.add(sha1(ir))

        if len(checksums) != 1:
            pytest.fail(
                f"Repeating the {action_name} action {i} times on "
                f"{benchmark_name} produced different states"
            )

        # An action which has no effect is not likely to be nondeterministic.
        if list(checksums)[0] == checksum_before:
            break


if __name__ == "__main__":
    main()
