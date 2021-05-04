# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from typing import List, Optional

import pytest

from compiler_gym.util import debug_util as dbg


def main(extra_pytest_args: Optional[List[str]] = None, debug_level: int = 1):
    """The main entry point for the pytest runner.

    An example file which uses this:

        from compiler_gym.util.test_main import main

        def test_foo():
            assert 1 + 1 == 2

        if __name__ == "__main__":
            main()

    In the above, the single test_foo test will be executed.

    :param extra_pytest_args: A list of additional command line options to pass
        to pytest.
    :param debug_level: The debug level to use to run tests. Higher levels are
        more verbose and may be useful for diagnosing test failures. Normally
        CompilerGym executes with a debug level of 0.
    """
    dbg.set_debug_level(debug_level)

    # Keep test data isolated from user data.
    os.environ["COMPILER_GYM_SITE_DATA"] = "/tmp/compiler_gym/tests/site_data"
    os.environ["COMPILER_GYM_CACHE"] = "/tmp/compiler_gym/tests/cache"

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

    pytest_args += extra_pytest_args or []

    returncode = pytest.main(pytest_args)

    # By default pytest will fail with an error if no tests are collected.
    # Disable that behavior here (with a warning) since there legitimate cases
    # where we may want to run a test file with no tests in it. For example,
    # when running on a continuous integration service where all the tests are
    # marked with the @skip_on_ci decorator.
    if returncode == pytest.ExitCode.NO_TESTS_COLLECTED.value:
        print(
            "WARNING: The test suite was empty. Is that intended?",
            file=sys.stderr,
        )
        returncode = 0

    sys.exit(returncode)
