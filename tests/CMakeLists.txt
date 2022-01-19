# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    compiler_env_test
  SRCS
    "compiler_env_test.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    compiler_env_state_test
  SRCS
    "compiler_env_state_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_test(
  NAME
    env_copy_test
  SRCS
    "env_copy_test.py"
  DEPS
    compiler_gym::envs::envs
    tests::pytest_plugins::llvm
    tests::test_main
)

cg_py_test(
  NAME
    make_test
  SRCS
    "make_test.py"
  DEPS
    compiler_gym::compiler_gym
    tests::test_main
)

cg_py_test(
  NAME
    random_search_test
  SRCS
    "random_search_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::random_replay
    compiler_gym::random_search
    tests::pytest_plugins::common
    tests::test_main
)

cg_py_library(
  NAME
    test_main
  SRCS
    "test_main.py"
  DEPS
    compiler_gym::util::util
  TESTONLY
  PUBLIC
)

cg_cc_library(
  NAME
    TestMacros
  SRCS
    "TestMacros.h"
  PUBLIC
)

cg_cc_library(
  NAME
    TestMain
  SRCS
    "TestMain.cc"
  ABS_DEPS
    glog::glog
    GTest::GTest
  TESTONLY
  PUBLIC
)

cg_py_test(
  NAME
    validate_test
  SRCS
    "validate_test.py"
  DEPS
    ::test_main
    compiler_gym::compiler_gym
)

cg_py_test(
  NAME
    validation_result_test
  SRCS
    "validation_result_test.py"
  DEPS
    ::test_main
    compiler_gym::compiler_gym
)

string(CONCAT _CMD
  "\"${CMAKE_COMMAND}\" -E create_symlink"
    " \"${CMAKE_CURRENT_SOURCE_DIR}/../VERSION\""
    " \"${CMAKE_CURRENT_BINARY_DIR}/../VERSION\"")
cg_genrule(
  NAME version_file
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION"
  OUTS "${CMAKE_CURRENT_BINARY_DIR}/../VERSION"
  COMMAND "${_CMD}"
)

cg_py_test(
  NAME
    version_test
  SRCS
    "version_test.py"
  DEPS
    ::test_main
    ::version_file
    compiler_gym::compiler_gym
    tests::pytest_plugins::common
)

file(GLOB_RECURSE _SRC_TEST_FILES
  LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}")
set(_TESTS_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/py_pkg")
string(CONCAT _CMD
  "cd \"${CMAKE_CURRENT_SOURCE_DIR}/..\""
  " && make install-test")
cg_genrule(
  NAME python_package_tests
  COMMAND "${_CMD}"
  DEPENDS compiler_gym::install_python_package
  EXCLUDE_FROM_ALL)
