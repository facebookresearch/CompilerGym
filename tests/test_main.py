# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

import gym
import pytest

import compiler_gym  # noqa Register environments.


def main():
    """The main entry point for the pytest runner.

    An example file which uses this:

        from compiler_gym.util.test_main import main

        def test_foo():
            assert 1 + 1 == 2

        if __name__ == "__main__":
            main()

    In the above, the single test_foo test will be executed.
    """
    # Use isolated data directories for running tests.
    os.environ["COMPILER_GYM_SITE_DATA"] = "/tmp/compiler_gym/tests/site_data"
    os.environ["COMPILER_GYM_CACHE"] = "/tmp/compiler_gym/tests/cache"

    # Install some benchmarks for the LLVM environment as otherwise
    # reset() will fail.
    env = gym.make("llvm-v0")
    try:
        env.require_dataset("cBench-v0")
    finally:
        env.close()

    # Use verbose backend debugging when running tests. If a test fails, the debugging
    # output will be included in the captured stderr.
    os.environ["COMPILER_GYM_SERVICE_DEBUG"] = "1"

    pytest_args = sys.argv + ["-vv"]
    # Support for sharding. If a py_test target has the shard_count attribute
    # set (in the range [1,50]), then the pytest-shard module is used to divide
    # the tests among the shards. See https://pypi.org/project/pytest-shard/
    sharded_test = os.environ.get("TEST_TOTAL_SHARDS")
    if sharded_test:
        num_shards = int(os.environ["TEST_TOTAL_SHARDS"])
        shard_index = int(os.environ["TEST_SHARD_INDEX"])
        pytest_args += [f"--shard-id={shard_index}", f"--num-shards={num_shards}"]
    else:
        pytest_args += ["-p", "no:pytest-shard"]

    sys.exit(pytest.main(pytest_args))
