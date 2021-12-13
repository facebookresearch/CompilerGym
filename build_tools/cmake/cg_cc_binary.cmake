# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_binary.cmake[
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# cg_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# TESTONLY: for testing; won't compile when tests are disabled
# HOSTONLY: host only; compile using host toolchain when cross-compiling
#
# Note:
# cg_cc_binary will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# cmake_base_foo with two alias (readonly) targets, a qualified
# ${PACKAGE_NS}::${NAME} and an unqualified ${NAME}. Thus NAME must be globally
# unique in the project.
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
# cg_cc_binary(
#   NAME
#     awesome_tool
#   SRCS
#     "awesome-tool-main.cc"
#   DEPS
#     compiler_gym::awesome
# )
function(cg_cc_binary)
  cmake_parse_arguments(
    _RULE
    "HOSTONLY;TESTONLY"
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;ABS_DEPS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT COMPILER_GYM_BUILD_TESTS)
    return()
  endif()

  cg_package_ns(_PACKAGE_NS)
  # Prefix the library with the package name, so we get: cg_package_name
  rename_bazel_targets(_NAME "${_RULE_NAME}")

  add_executable(${_NAME} "")
  add_executable(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  # If the binary name matches the package then treat it as a default. For
  # example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
  # likely to be common for binaries, but is consistent with the behavior for
  # libraries and in Bazel.
  cg_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    add_executable(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()

  # Finally, since we have so few binaries and we also want to support
  # installing from a separate host build, binaries get an unqualified global
  # alias. This means binary names must be unique across the whole project.
  # (We could consider making this configurable).
  add_executable(${_RULE_NAME} ALIAS ${_NAME})

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${COMPILER_GYM_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${COMPILER_GYM_BINARY_DIR}>"
  )
  target_include_directories(${_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${COMPILER_GYM_DEFAULT_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${COMPILER_GYM_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  rename_bazel_targets(_RULE_DEPS "${_RULE_DEPS}")

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
      ${_RULE_ABS_DEPS}
  )

  cg_add_data_dependencies(NAME ${_RULE_NAME} DATA ${_RULE_DATA})

  # Add all targets to a folder in the IDE for organization.
  set_target_properties(${_NAME} PROPERTIES
    FOLDER ${COMPILER_GYM_IDE_FOLDER}/binaries
    CXX_STANDARD ${COMPILER_GYM_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON)

  install(TARGETS ${_NAME}
          RENAME ${_RULE_NAME}
          COMPONENT ${_RULE_NAME}
          RUNTIME DESTINATION bin)
endfunction()
