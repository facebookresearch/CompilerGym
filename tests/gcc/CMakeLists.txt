# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    gcc_bin_test
  SRCS
    "gcc_bin_test.py"
  DEPS
    compiler_gym::envs::gcc::gcc
    compiler_gym::service::service
    tests::pytest_plugins::gcc
    tests::test_main
)

cg_py_test(
  NAME
    gcc_docker_test
  SRCS
    "gcc_docker_test.py"
  DEPS
    compiler_gym::envs::gcc::gcc
    compiler_gym::service::service
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    gcc_env_test
  SRCS
    "gcc_env_test.py"
  DEPS
    compiler_gym::envs::gcc::gcc
    compiler_gym::service::service
    compiler_gym::spaces::spaces
    tests::pytest_plugins::common
    tests::pytest_plugins::gcc
    tests::test_main
)
