# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/iree_cc_library.cmake
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# cg_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DCOMPILER_GYM_BUILD_TESTS=ON to CMake.
# SHARED: If set, will compile to a shared object.
#
# cg_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# cg_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     package::awesome # not "awesome" !
#   PUBLIC
# )
#
# cg_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     package::fantastic_lib
# )
function(cg_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;SHARED"
    "NAME"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;ABS_DEPS;NON_LIB_DEPS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT COMPILER_GYM_BUILD_TESTS)
    return()
  endif()

  cg_package_ns(_PACKAGE_NS)
  rename_bazel_targets(_DEPS "${_RULE_DEPS}")
  list(APPEND _DEPS ${_RULE_ABS_DEPS})

  # Prefix the library with the package name, so we get: cg_package_name.
  rename_bazel_targets(_NAME "${_RULE_NAME}")

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  if(NOT _RULE_IS_INTERFACE)
    if(_RULE_SHARED)
      add_library(${_NAME} SHARED "")
    else()
      add_library(${_NAME} STATIC "")
    endif()
    if(_RULE_SRCS)
      list(JOIN _RULE_SRCS ";\n" SRCSTR)
      message("${SRCSTR}")
    endif()
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )
    target_include_directories(${_NAME} SYSTEM
      PUBLIC
        "$<BUILD_INTERFACE:${COMPILER_GYM_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${COMPILER_GYM_BINARY_DIR}>"
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
        ${_RULE_INCLUDES}
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
    target_link_libraries(${_NAME}
      PUBLIC
        ${_DEPS}
    )

    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    # Add all targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${COMPILER_GYM_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${COMPILER_GYM_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${COMPILER_GYM_IDE_FOLDER}/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set.
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${COMPILER_GYM_CXX_STANDARD})
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library.
    add_library(${_NAME} INTERFACE ${_RULE_SRCS} ${_RULE_TEXTUAL_HDRS} ${_RULE_HDRS})
    target_include_directories(${_NAME} SYSTEM
      INTERFACE
        "$<BUILD_INTERFACE:${COMPILER_GYM_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${COMPILER_GYM_BINARY_DIR}>"
    )
    target_link_options(${_NAME}
      INTERFACE
        ${COMPILER_GYM_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_DEPS}
    )
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  cg_add_data_dependencies(NAME ${_RULE_NAME} DATA ${_RULE_DATA})

  if (_RULE_NON_LIB_DEPS)
    rename_bazel_targets(_NON_LIB_DEPS "${_RULE_NON_LIB_DEPS}")
    add_dependencies(${_NAME} ${_NON_LIB_DEPS})
  endif()

  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  # If the library name matches the final component of the package then treat
  # it as a default. For example, foo/bar/ library 'bar' would end up as
  # 'foo::bar'.
  cg_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()
endfunction()
