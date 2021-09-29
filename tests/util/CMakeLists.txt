# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    capture_output_test
  SRCS
    "capture_output_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME
    debug_util_test
  SRCS
    "debug_util_test.py"
  DEPS
    compiler_gym::util::util
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    download_test
  SRCS
    "download_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME executor_test
  SRCS executor_test.py
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_cc_test(
  NAME
    EnumUtilTest
  SRCS
    "EnumUtilTest.cc"
  DEPS
    compiler_gym::util::EnumUtil
    tests::TestMacros
    tests::TestMain
  ABS_DEPS
    GTest::GTest
)

cg_py_test(
  NAME
    filesystem_test
  SRCS
    "filesystem_test.py"
  DEPS
    compiler_gym::util::util
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    minimize_trajectory_test
  SRCS
    "minimize_trajectory_test.py"
  DEPS
    compiler_gym::util::util
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    parallelization_test
  SRCS
    "parallelization_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME runfiles_path_test
  SRCS "runfiles_path_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
    tests::pytest_plugins::common
)

cg_py_test(
  NAME shell_format_test
  SRCS shell_format_test.py
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME
    statistics_test
  SRCS
    "statistics_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_cc_test(
  NAME
    StrLenConstexprTest
  SRCS
    "StrLenConstexprTest.cc"
  DEPS
    compiler_gym::util::StrLenConstexpr
    tests::TestMain
)

cg_cc_test(
  NAME
    SubprocessTest
  SRCS
    "SubprocessTest.cc"
  DEPS
    compiler_gym::util::Subprocess
    tests::TestMain
)

cg_py_test(
  NAME
    temporary_working_directory_test
  SRCS
    "temporary_working_directory_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME
    timer_test
  SRCS
    "timer_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)

cg_py_test(
  NAME
    truncate_test
  SRCS
    "truncate_test.py"
  DEPS
    compiler_gym::util::util
    tests::test_main
)
