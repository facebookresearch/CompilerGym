# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)
include(CMakeParseArguments)

#-------------------------------------------------------------------------------
# Missing CMake Variables
#-------------------------------------------------------------------------------

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(COMPILER_GYM_HOST_SCRIPT_EXT "bat")
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(COMPILER_GYM_HOST_EXECUTABLE_SUFFIX ".exe")
else()
  set(COMPILER_GYM_HOST_SCRIPT_EXT "sh")
  set(COMPILER_GYM_HOST_EXECUTABLE_SUFFIX "")
endif()

#-------------------------------------------------------------------------------
# General utilities
#-------------------------------------------------------------------------------

# cg_to_bool
#
# Sets `variable` to `ON` if `value` is true and `OFF` otherwise.
function(cg_to_bool VARIABLE VALUE)
  if(VALUE)
    set(${VARIABLE} "ON" PARENT_SCOPE)
  else()
    set(${VARIABLE} "OFF" PARENT_SCOPE)
  endif()
endfunction()

# cg_append_list_to_string
#
# Joins ${ARGN} together as a string separated by " " and appends it to
# ${VARIABLE}.
function(cg_append_list_to_string VARIABLE)
  if(NOT "${ARGN}" STREQUAL "")
    string(JOIN " " _ARGN_STR ${ARGN})
    set(${VARIABLE} "${${VARIABLE}} ${_ARGN_STR}" PARENT_SCOPE)
  endif()
endfunction()


#-------------------------------------------------------------------------------
# Packages and Paths
#-------------------------------------------------------------------------------

# Sets ${PACKAGE_NS} to the root relative package name in C++ namespace
# format (::).
#
# Example when called from proj/base/CMakeLists.txt:
#   proj::base
function(cg_package_ns PACKAGE_NS)
  string(REPLACE "${COMPILER_GYM_ROOT_DIR}" "" _PACKAGE "${CMAKE_CURRENT_LIST_DIR}")
  string(LENGTH "${_PACKAGE}" _LENGTH)
  if (_LENGTH)
    string(SUBSTRING "${_PACKAGE}" 1 -1 _PACKAGE)
  endif()
  string(REPLACE "/" "::" _PACKAGE_NS "${_PACKAGE}")
  set(${PACKAGE_NS} "${_PACKAGE_NS}" PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the root relative package name.
#
# Example when called from proj/base/CMakeLists.txt:
#   proj__base
function(cg_package_name PACKAGE_NAME)
  cg_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "__" _PACKAGE_NAME ${_PACKAGE_NS})
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the root relative package path.
#
# Example when called from proj/base/CMakeLists.txt:
#   proj/base
function(cg_package_path PACKAGE_PATH)
  cg_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_DIR} to the directory name of the current package.
#
# Example when called from proj/base/CMakeLists.txt:
#   base
function(cg_package_dir PACKAGE_DIR)
  cg_package_ns(_PACKAGE_NS)
  string(FIND ${_PACKAGE_NS} "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

function(canonize_bazel_target_names _RESULT _BAZEL_TARGETS)
  unset(_RES)
  cg_package_ns(_PACKAGE_NS)
  foreach(_TARGET ${_BAZEL_TARGETS})
    if (NOT _TARGET MATCHES ":")
      # local target
      if (NOT _PACKAGE_NS STREQUAL "")
        set(_TARGET "${_PACKAGE_NS}::${_TARGET}")
      endif()
    endif()
    list(APPEND _RES "${_TARGET}")
  endforeach()
  list(TRANSFORM _RES REPLACE "^::" "${_PACKAGE_NS}::")
  set(${_RESULT} ${_RES} PARENT_SCOPE)
endfunction()

function(rename_bazel_targets _RESULT _BAZEL_TARGETS)
  canonize_bazel_target_names(_RES "${_BAZEL_TARGETS}")
  list(TRANSFORM _RES REPLACE ":" "_")
  set(${_RESULT} ${_RES} PARENT_SCOPE)
endfunction()

function(get_target_as_relative_dir _TARGET _RESULT)
  set(_RES "${_TARGET}")
  list(TRANSFORM _RES REPLACE "__" "/")
  get_filename_component(_RES "${_RES}" DIRECTORY)
  set(${_RESULT} "${_RES}" PARENT_SCOPE)
endfunction()

function(get_target_out_cxx_header_dir _TARGET _RESULT)
  get_target_property(_BIN_DIR ${_TARGET} BINARY_DIR)
  get_target_as_relative_dir(${_TARGET} _REL_HEADER_DIR)
  set(${_RESULT} "${_BIN_DIR}/include/${_REL_HEADER_DIR}" PARENT_SCOPE)
endfunction()

function(make_paths_absolute)
  cmake_parse_arguments(
    _ARG
    ""
    "BASE_DIR;RESULT_VARIABLE"
    "PATHS"
    ${ARGN}
  )

  unset(_RES)
  foreach(_PATH ${_ARG_PATHS})
    if(NOT IS_ABSOLUTE _PATH)
      get_filename_component(_PATH "${_PATH}" ABSOLUTE BASE_DIR "${_ARG_BASE_DIR}")
    endif()
    list(APPEND _RES "${_PATH}")
  endforeach()

  set(${_ARG_RESULT_VARIABLE} "${_RES}" PARENT_SCOPE)
endfunction()

function(paths_to_targets)
  cmake_parse_arguments(
    _ARG
    ""
    "RESULT"
    "PATHS"
    ${ARGN}
  )

  string(REGEX REPLACE "[^A-Za-z0-9_+-]" "_" _TARGETS "${_ARG_PATHS}")
  set(${_ARG_RESULT} "${_TARGETS}" PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# select()-like Evaluation
#-------------------------------------------------------------------------------

# Appends ${OPTS} with a list of values based on the current compiler.
#
# Example:
#   cg_select_compiler_opts(COPTS
#     CLANG
#       "-Wno-foo"
#       "-Wno-bar"
#     CLANG_CL
#       "/W3"
#     GCC
#       "-Wsome-old-flag"
#     MSVC
#       "/W3"
#   )
#
# Note that variables are allowed, making it possible to share options between
# different compiler targets.
function(cg_select_compiler_opts OPTS)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _COMPILER_GYM_SELECTS
    ""
    ""
    "ALL;CLANG;CLANG_CL;MSVC;GCC;CLANG_OR_GCC;MSVC_OR_CLANG_CL"
  )
  # OPTS is a variable containing the *name* of the variable being populated, so
  # we need to dereference it twice.
  set(_OPTS "${${OPTS}}")
  list(APPEND _OPTS "${_COMPILER_GYM_SELECTS_ALL}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND _OPTS "${_COMPILER_GYM_SELECTS_GCC}")
    list(APPEND _OPTS "${_COMPILER_GYM_SELECTS_CLANG_OR_GCC}")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(MSVC)
      list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_CLANG_CL})
      list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_MSVC_OR_CLANG_CL})
    else()
      list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_CLANG})
      list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_CLANG_OR_GCC})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_MSVC})
    list(APPEND _OPTS ${_COMPILER_GYM_SELECTS_MSVC_OR_CLANG_CL})
  else()
    message(ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER}")
    list(APPEND _OPTS "")
  endif()
  set(${OPTS} ${_OPTS} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Data dependencies
#-------------------------------------------------------------------------------

# Adds 'data' dependencies to a target.
#
# Parameters:
# NAME: name of the target to add data dependencies to
# DATA: List of targets and/or files in the source tree. Files should use the
#       same format as targets (i.e. iree::package::subpackage::file.txt)
function(cg_add_data_dependencies)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "DATA"
    ${ARGN}
  )
  # TODO(boian): Make runtime targets that depend on data

  if(NOT DEFINED _RULE_DATA)
    return()
  endif()

  rename_bazel_targets(_NAME "${_RULE_NAME}")
  unset(_DEPS)

  foreach(_DATA ${_RULE_DATA})
    if(IS_ABSOLUTE "${_DATA}")
      get_filename_component(FILE_ "${_DATA}" ABSOLUTE)
      paths_to_targets(PATHS "${FILE_}" RESULT _TARGET)
      string(PREPEND _TARGET "${_NAME}_data_")
      get_filename_component(_FILE_NAME "${FILE_}" NAME)
      set(_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
      set(_DST_PATH "${_DST_DIR}/${_FILE_NAME}")
      if(NOT _DST_PATH STREQUAL _DATA)
        add_custom_command(
          OUTPUT "${_DST_PATH}"
          COMMAND
            ${CMAKE_COMMAND} -E make_directory "${_DST_DIR}"
          COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${FILE_}" "${_DST_PATH}"
          DEPENDS "${FILE_}"
          VERBATIM
        )
      endif()
      add_custom_target(${_TARGET} DEPENDS "${_DST_PATH}")
    else()
      rename_bazel_targets(_TARGET "${_DATA}")
    endif()
    list(APPEND _DEPS "${_TARGET}")
  endforeach()

  add_dependencies(${_NAME} ${_DEPS})
endfunction()

#-------------------------------------------------------------------------------
# Tool symlinks
#-------------------------------------------------------------------------------

# cg_symlink_tool
#
# Adds a command to TARGET which symlinks a tool from elsewhere
# (FROM_TOOL_TARGET_NAME) to a local file name (TO_EXE_NAME) in the current
# binary directory.
#
# Parameters:
#   TARGET: Local target to which to add the symlink command (i.e. an
#     cg_py_library, etc).
#   FROM_TOOL_TARGET: Target of the tool executable that is the source of the
#     link.
#   TO_EXE_NAME: The executable name to output in the current binary dir.
function(cg_symlink_tool)
  cmake_parse_arguments(
    ARG
    ""
    "TARGET;FROM_TOOL_TARGET;TO_EXE_NAME"
    ""
    ${ARGN}
  )

  # Transform TARGET
  cg_package_ns(_PACKAGE_NS)
  cg_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${ARG_TARGET}")
  set(_FROM_TOOL_TARGET ${ARG_FROM_TOOL_TARGET})
  set(_TO_TOOL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
  get_filename_component(_TO_TOOL_DIR "${_TO_TOOL_PATH}" DIRECTORY)


  add_custom_command(
    TARGET "${_TARGET}"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory "${_TO_TOOL_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${_FROM_TOOL_TARGET}>"
        "${_TO_TOOL_PATH}"
    VERBATIM
  )
endfunction()


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

# cg_add_test_environment_properties
#
# Adds test environment variable properties based on the current build options.
#
function(cg_add_test_environment_properties TEST_NAME)
  # COMPILER_GYM_*_DISABLE environment variables may used to skip test cases which
  # require both a compiler target backend and compatible runtime HAL driver.
  #
  # These variables may be set by the test environment, typically as a property
  # of some continuous execution test runner or by an individual developer, or
  # here by the build system.
  #
  # Tests which only depend on a compiler target backend or a runtime HAL
  # driver, but not both, should generally use a different method of filtering.
  if(NOT "${COMPILER_GYM_TARGET_BACKEND_VULKAN-SPIRV}" OR NOT "${COMPILER_GYM_HAL_DRIVER_VULKAN}")
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "COMPILER_GYM_VULKAN_DISABLE=1")
  endif()
  if(NOT "${COMPILER_GYM_TARGET_BACKEND_DYLIB-LLVM-AOT}" OR NOT "${COMPILER_GYM_HAL_DRIVER_DYLIB}"
     OR NOT "${COMPILER_GYM_HAL_DRIVER_DYLIB_SYNC}")
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "COMPILER_GYM_LLVMAOT_DISABLE=1")
  endif()
endfunction()
