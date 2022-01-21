# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    manual_env_bin_test
  SRCS
    "manual_env_bin_test.py"
  DEPS
    compiler_gym::bin::manual_env
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME
    service_bin_test
  SRCS
    "service_bin_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::bin::service
    tests::test_main
)

cg_py_test(
  NAME
    validate_bin_test
  SRCS
    "validate_bin_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::bin::validate
    tests::pytest_plugins::common
    tests::test_main
)
