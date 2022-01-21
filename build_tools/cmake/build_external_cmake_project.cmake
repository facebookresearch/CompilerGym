# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CMakeParseArguments)
include(write_cache_script)

function(build_external_cmake_project)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC_DIR;INSTALL_PREFIX"
    "CONFIG_ARGS"
    ${ARGN}
  )

  set(_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/${_RULE_NAME}")
  if(_RULE_INSTALL_PREFIX)
    set(_INSTALL_PREFIX "${_RULE_INSTALL_PREFIX}")
  else()
    set(_INSTALL_PREFIX "${_BIN_DIR}/install")
  endif()

  set(_INTIAL_CACHE_PATH "${_BIN_DIR}/${_RULE_NAME}_initial_cache.cmake")
  write_cache_script("${_INTIAL_CACHE_PATH}")

  execute_process(
    COMMAND "${CMAKE_COMMAND}"
    -G "${CMAKE_GENERATOR}" # For some reason the generator is not taken from the initial cache.
    -C "${_INTIAL_CACHE_PATH}"
    -S "${_RULE_SRC_DIR}"
    -B "${_BIN_DIR}"
    -D "CMAKE_INSTALL_PREFIX=${_INSTALL_PREFIX}"
    ${_RULE_CONFIG_ARGS}
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
    "${CMAKE_COMMAND}"
    --build "${_BIN_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  execute_process(
    COMMAND
    "${CMAKE_COMMAND}"
    --install "${_BIN_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  list(PREPEND CMAKE_PREFIX_PATH "${_INSTALL_PREFIX}")
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
endfunction()
