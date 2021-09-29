# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    anghabench_test
  SRCS
    "anghabench_test.py"
  DEPS
    compiler_gym::envs::gcc::gcc
    compiler_gym::envs::gcc::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::gcc
    tests::test_main
)

cg_py_test(
  NAME
    csmith_test
  SRCS
    "csmith_test.py"
  DEPS
    compiler_gym::envs::gcc::gcc
    compiler_gym::envs::gcc::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::gcc
    tests::test_main
)
