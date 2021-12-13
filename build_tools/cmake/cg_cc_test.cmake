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

# cg_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target. This name is used for the generated executable and
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
#
# Note:
# cg_cc_test will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# cg_base_foo_test.
#
#
# Usage:
# cg_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# cg_cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     gtest_main
#     compiler_gym::awesome
# )
function(cg_cc_test)
  if(NOT COMPILER_GYM_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;LABELS"
    ${ARGN}
  )

  cg_cc_binary(${ARGV})

  rename_bazel_targets(_NAME "${_RULE_NAME}")
  cg_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")
  set(_LABELS "${_RULE_LABELS}")
  list(APPEND _LABELS "${_PACKAGE_PATH}")

  cg_add_installed_test(
    TEST_NAME "${_TEST_NAME}"
    LABELS "${_LABELS}"
    COMMAND
      # We run all our tests through a custom test runner to allow temp
      # directory cleanup upon test completion.
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${COMPILER_GYM_HOST_SCRIPT_EXT}"
      "$<TARGET_FILE:${_NAME}>"
    INSTALLED_COMMAND
      # Must match install destination below.
      "${_PACKAGE_PATH}/$<TARGET_FILE_NAME:${_NAME}>"
  )

  install(TARGETS ${_NAME}
    DESTINATION "tests/${_PACKAGE_PATH}"
    COMPONENT Tests
  )

endfunction()
