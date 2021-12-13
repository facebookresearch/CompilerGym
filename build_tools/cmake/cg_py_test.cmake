# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_test.cmake
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)
include(cg_installed_test)

# cg_py_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target.
# SRCS: List of source files
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# ARGS command line arguments for the test.
#
# Note:
# cg_cc_test will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# cg_base_foo_test.
#
function(cg_py_test)
  if(NOT COMPILER_GYM_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRCS"
    "ARGS;LABELS;DATA;DEPS"
    ${ARGN}
  )

  cg_py_binary(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    DATA ${_RULE_DATA}
  )

  rename_bazel_targets(_NAME "${_RULE_NAME}")
  cg_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  set(_LABELS "${_RULE_LABELS}")
  list(APPEND _LABELS "${_PACKAGE_PATH}")

  cg_add_installed_test(
    TEST_NAME "${_TEST_NAME}"
    LABELS "${_LABELS}"
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}"
      "TEST_WORKSPACE=compiler_gym"
      #"COMPILER_GYM_RUNFILES=${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${COMPILER_GYM_HOST_SCRIPT_EXT}"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRCS}"
      ${_RULE_ARGS}
    INSTALLED_COMMAND
      python
      "${_PACKAGE_PATH}/${_RULE_SRCS}"
  )

  #cg_add_data_dependencies(NAME ${_RULE_NAME} DATA ${_RULE_DATA})

  install(FILES ${_RULE_SRCS}
    DESTINATION "tests/${_PACKAGE_PATH}"
    COMPONENT Tests
  )

  # TODO(boian): Find out how to add deps to tests.
  # CMake seems to not allow build targets to be dependencies for tests.
  # One way to achieve this is to make the test execution a target.
endfunction()
