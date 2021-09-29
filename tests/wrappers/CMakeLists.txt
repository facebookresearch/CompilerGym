# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    commandline_wrappers_test
  SRCS
    "commandline_wrappers_test.py"
  DEPS
    compiler_gym::wrappers::wrappers
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    core_wrappers_test
  SRCS
    "core_wrappers_test.py"
  DEPS
    compiler_gym::wrappers::wrappers
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    datasets_wrappers_test
  SRCS
    "datasets_wrappers_test.py"
  DEPS
    compiler_gym::wrappers::wrappers
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME llvm_test
  SRCS "llvm_test.py"
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::wrappers::wrappers
    tests::test_main
    tests::pytest_plugins::llvm
)

cg_py_test(
  NAME
    time_limit_wrappers_test
  SRCS
    "time_limit_wrappers_test.py"
  DEPS
    compiler_gym::wrappers::wrappers
    tests::pytest_plugins::llvm
    tests::test_main
)
