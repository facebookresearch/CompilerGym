# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CMakeParseArguments)
include(cg_macros)
include(cg_py_library)

function(proto_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC"
    "NAME;SRCS"
    "DEPS"
    ${ARGN}
  )

  rename_bazel_targets(_RULE_DEPS "${_RULE_DEPS}")
  rename_bazel_targets(_RULE_NAME "${_RULE_NAME}")

  set(_SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}")
  set(_DST_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRCS}.bin")
  get_filename_component(_DST_DIR "${_DST_FILE}" DIRECTORY)
  file(RELATIVE_PATH _RELATIVE_PROTO_FILE "${CMAKE_SOURCE_DIR}" "${_SRC_FILE}")

  add_custom_command(
    OUTPUT "${_DST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --proto_path "${CMAKE_SOURCE_DIR}"
      --descriptor_set_out "${_DST_FILE}"
      "${_RELATIVE_PROTO_FILE}"
    DEPENDS "${Protobuf_PROTOC_EXECUTABLE}" "${_SRC_FILE}" ${_RULE_DEPS}
    VERBATIM)

  add_custom_target(${_RULE_NAME} ALL DEPENDS "${_DST_FILE}")
  set_target_properties(${_RULE_NAME} PROPERTIES PROTO_DESCRIPTOR_SETS "${_DST_FILE}")
  set_target_properties(${_RULE_NAME} PROPERTIES PROTO_FILES "${_SRC_FILE}")
endfunction()

function(get_cc_proto_out_files _PROTO_FILENAME _RESULT)
  set(_PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto\\.bin$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME_WITHOUT_EXT}")
  set(${_RESULT}
    "${_PROTO_FILENAME_WITHOUT_EXT}.pb.h"
    "${_PROTO_FILENAME_WITHOUT_EXT}.pb.cc"
    PARENT_SCOPE)
endfunction()

function(cc_proto_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC"
    "NAME;DEPS"
    ""
    ${ARGN}
  )

  rename_bazel_targets(_DEPS "${_RULE_DEPS}")
  rename_bazel_targets(_NAME "${_RULE_NAME}")

  get_target_as_relative_dir(${_NAME} _HEADER_DST_DIR)
  set(_HEADER_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
  get_target_property(_DESCRIPTOR_SET_FILE ${_DEPS} PROTO_DESCRIPTOR_SETS)

  get_target_property(_PROTO_FILE ${_DEPS} PROTO_FILES)
  get_filename_component(_PROTO_FILENAME "${_PROTO_FILE}" NAME)
  file(RELATIVE_PATH _RELATIVE_PROTO_FILE "${CMAKE_SOURCE_DIR}" "${_PROTO_FILE}")

  get_filename_component(_RELATIVE_PROTO_DIR "${_RELATIVE_PROTO_FILE}" DIRECTORY)
  get_filename_component(_SRC_FILENAME "${_DESCRIPTOR_SET_FILE}" NAME)
  get_cc_proto_out_files("${_SRC_FILENAME}" _CC_PROTO_FILES)
  list(TRANSFORM _CC_PROTO_FILES PREPEND "${_HEADER_DST_DIR}/${_RELATIVE_PROTO_DIR}/")

  add_custom_command(
    OUTPUT ${_CC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_HEADER_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --proto_path "${CMAKE_SOURCE_DIR}"
      --descriptor_set_in "${_DESCRIPTOR_SET_FILE}"
      --cpp_out "${_HEADER_DST_DIR}"
      "${_RELATIVE_PROTO_FILE}"
    DEPENDS
      "${Protobuf_PROTOC_EXECUTABLE}"
      "${_DESCRIPTOR_SET_FILE}"
      "${_PROTO_FILE}"
      ${_DEPS}
    VERBATIM)

  cg_cc_library(
    NAME ${_RULE_NAME}
    SRCS ${_CC_PROTO_FILES}
    ABS_DEPS protobuf::libprotobuf
    INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/include"
    PUBLIC
  )
endfunction()

function(get_py_proto_out_files _PROTO_FILENAME _RESULT)
  set(_PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto\\.bin$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME_WITHOUT_EXT}")
  set(${_RESULT}
    "${_PROTO_FILENAME_WITHOUT_EXT}_pb2.py"
    PARENT_SCOPE)
endfunction()

function(py_proto_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC"
    "NAME;DEPS"
    ""
    ${ARGN}
  )

  rename_bazel_targets(_DEPS "${_RULE_DEPS}")

  get_target_property(_DESCRIPTOR_SET_FILE ${_DEPS} PROTO_DESCRIPTOR_SETS)
  get_filename_component(_SRC_FILENAME "${_DESCRIPTOR_SET_FILE}" NAME)
  get_py_proto_out_files("${_SRC_FILENAME}" _PY_PROTO_FILES)
  set(_PYTHON_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(_ABS_PATH_PY_PROTO_FILES ${_PY_PROTO_FILES})
  list(TRANSFORM _ABS_PATH_PY_PROTO_FILES PREPEND "${_PYTHON_DST_DIR}/")

  get_target_property(_PROTO_FILE ${_DEPS} PROTO_FILES)
  file(RELATIVE_PATH _RELATIVE_PROTO_FILE "${CMAKE_SOURCE_DIR}" "${_PROTO_FILE}")

  add_custom_command(
    OUTPUT ${_ABS_PATH_PY_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --proto_path "${CMAKE_SOURCE_DIR}"
      --descriptor_set_in "${_DESCRIPTOR_SET_FILE}"
      --python_out "${CMAKE_BINARY_DIR}"
      "${_RELATIVE_PROTO_FILE}"
    DEPENDS
      "${Protobuf_PROTOC_EXECUTABLE}"
      "${_DESCRIPTOR_SET_FILE}"
      "${_PROTO_FILE}"
      ${_DEPS}
    VERBATIM)

  cg_py_library(
    NAME "${_RULE_NAME}"
    GENERATED_SRCS ${_PY_PROTO_FILES}
    )
endfunction()
