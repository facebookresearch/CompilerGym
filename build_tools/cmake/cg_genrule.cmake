# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

include(CMakeParseArguments)
include(cg_macros)

# cg_genrule()
#
# CMake function to imitate Bazel's genrule rule.
#
function(cg_genrule)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;EXCLUDE_FROM_ALL"
    "NAME;COMMAND"
    "SRCS;OUTS;DEPENDS;ABS_DEPENDS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT COMPILER_GYM_BUILD_TESTS)
    return()
  endif()

  # TODO(boian): remove this renaming when call sites do not include ":" in target dependency names
  rename_bazel_targets(_DEPS "${_RULE_DEPENDS}")

  rename_bazel_targets(_NAME "${_RULE_NAME}")

  make_paths_absolute(
    PATHS ${_RULE_SRCS}
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE _SRCS
  )

  make_paths_absolute(
    PATHS ${_RULE_OUTS}
    BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    RESULT_VARIABLE _OUTS
  )

  list(LENGTH _OUTS _OUTS_LENGTH)
  if(_OUTS_LENGTH EQUAL 1)
    get_filename_component(_OUTS_DIR "${_OUTS}" DIRECTORY)
  else()
    set(_OUTS_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  # Substitute special Bazel references
  string(REPLACE  "$@" "${_OUTS}" _CMD "${_RULE_COMMAND}")
  string(REPLACE  "$(@D)" "${_OUTS_DIR}" _CMD "${_CMD}")

  if(_OUTS)
    add_custom_command(
      OUTPUT ${_OUTS}
      COMMAND bash -c "${_CMD}"
      DEPENDS ${_DEPS} ${_SRCS}
      VERBATIM
      USES_TERMINAL
    )
  endif()

  if(_RULE_EXCLUDE_FROM_ALL)
    unset(_ALL)
  else()
    set(_ALL ALL)
  endif()

  if(_OUTS)
    add_custom_target(${_NAME} ${_ALL} DEPENDS ${_OUTS})
  else()
    add_custom_target(
      ${_NAME} ${_ALL}
      COMMAND bash -c "${_CMD}"
      DEPENDS ${_DEPS} ${_SRCS}
      VERBATIM
      USES_TERMINAL)
  endif()

  set_target_properties(${_NAME} PROPERTIES
    OUTPUTS "${_OUTS}")

  list(LENGTH _OUTS _OUTS_LENGTH)
  if(_OUTS_LENGTH EQUAL "1")
    set_target_properties(${_NAME} PROPERTIES LOCATION "${_OUTS}")
  endif()

endfunction()
