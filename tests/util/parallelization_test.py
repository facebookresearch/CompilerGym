# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:parallelization."""
from compiler_gym.util import parallelization
from tests.test_main import main


def test_thread_safe_tee():
    a, b = parallelization.thread_safe_tee(range(100))
    assert next(a) == 0
    assert next(b) == 0


if __name__ == "__main__":
    main()
