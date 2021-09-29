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
include(protobuf)

function(get_cc_grpc_proto_out_files _PROTO_FILENAME _RESULT)
  set(_PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto\\.bin$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME_WITHOUT_EXT}")
  set(${_RESULT}
    "${_PROTO_FILENAME_WITHOUT_EXT}.grpc.pb.h"
    "${_PROTO_FILENAME_WITHOUT_EXT}.grpc.pb.cc"
    PARENT_SCOPE)
endfunction()

function(cc_grpc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;GRPC_ONLY"
    "NAME;SRCS"
    "DEPS"
    ${ARGN}
  )

  if (NOT _RULE_GRPC_ONLY)
    message("GRPC_ONLY=False unsupported.")
  endif()

  rename_bazel_targets(_DEPS "${_RULE_DEPS}")
  rename_bazel_targets(_NAME "${_RULE_NAME}")
  rename_bazel_targets(_SRCS "${_RULE_SRCS}")

  get_target_as_relative_dir(${_NAME} _HEADER_DST_DIR)
  set(_HEADER_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/")
  get_target_property(_DESCRIPTOR_SET_FILE ${_SRCS} PROTO_DESCRIPTOR_SETS)

  get_target_property(_PROTO_FILE ${_SRCS} PROTO_FILES)
  file(RELATIVE_PATH _RELATIVE_PROTO_FILE "${CMAKE_SOURCE_DIR}" "${_PROTO_FILE}")

  get_filename_component(_RELATIVE_PROTO_DIR "${_RELATIVE_PROTO_FILE}" DIRECTORY)
  get_filename_component(_SRC_FILENAME "${_DESCRIPTOR_SET_FILE}" NAME)
  get_cc_grpc_proto_out_files("${_SRC_FILENAME}" _GRPC_PROTO_FILES)
  list(TRANSFORM _GRPC_PROTO_FILES PREPEND "${_HEADER_DST_DIR}/${_RELATIVE_PROTO_DIR}/")

  add_custom_command(
    OUTPUT ${_GRPC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_HEADER_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --proto_path "${CMAKE_SOURCE_DIR}"
      --descriptor_set_in "${_DESCRIPTOR_SET_FILE}"
      --grpc_out "${_HEADER_DST_DIR}"
      --plugin "protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${_RELATIVE_PROTO_FILE}"
    DEPENDS "${Protobuf_PROTOC_EXECUTABLE}" "${_DESCRIPTOR_SET_FILE}" "${_PROTO_FILE}" ${_DEPS}
    VERBATIM)

  cg_cc_library(
    NAME ${_RULE_NAME}
    SRCS ${_GRPC_PROTO_FILES}
    ABS_DEPS grpc++
    INCLUDES "${CMAKE_CURRENT_BINARY_DIR}/include"
    PUBLIC
  )
endfunction()

function(get_py_grpc_proto_out_files _PROTO_FILENAME _RESULT)
  set(_PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME}")
  string(REGEX REPLACE "\\.proto\\.bin$" "" _PROTO_FILENAME_WITHOUT_EXT "${_PROTO_FILENAME_WITHOUT_EXT}")
  set(${_RESULT}
    "${_PROTO_FILENAME_WITHOUT_EXT}_pb2_grpc.py"
    PARENT_SCOPE)
endfunction()

function(py_grpc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRCS"
    "DEPS"
    ${ARGN}
  )

  rename_bazel_targets(_DEPS "${_RULE_DEPS}")
  rename_bazel_targets(_SRCS "${_RULE_SRCS}")

  get_target_property(_DESCRIPTOR_SET_FILE ${_SRCS} PROTO_DESCRIPTOR_SETS)
  get_filename_component(_SRC_FILENAME "${_DESCRIPTOR_SET_FILE}" NAME)
  get_py_grpc_proto_out_files("${_SRC_FILENAME}" _PY_GRPC_PROTO_FILES)
  set(_PYTHON_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(_ABS_PATH_PY_GRPC_PROTO_FILES ${_PY_GRPC_PROTO_FILES})
  list(TRANSFORM _ABS_PATH_PY_GRPC_PROTO_FILES PREPEND "${_PYTHON_DST_DIR}/")

  get_target_property(_PROTO_FILE ${_SRCS} PROTO_FILES)
  get_filename_component(_PROTO_FILENAME "${_PROTO_FILE}" NAME)
  file(RELATIVE_PATH _RELATIVE_PROTO_FILE "${CMAKE_SOURCE_DIR}" "${_PROTO_FILE}")

  add_custom_command(
    OUTPUT ${_ABS_PATH_PY_GRPC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_PYTHON_DST_DIR}"
    COMMAND "${Python3_EXECUTABLE}"
      -m grpc_tools.protoc
      --proto_path "${CMAKE_SOURCE_DIR}"
      --descriptor_set_in "${_DESCRIPTOR_SET_FILE}"
      --grpc_python_out "${CMAKE_BINARY_DIR}"
      "${_RELATIVE_PROTO_FILE}"
    DEPENDS "${Python3_EXECUTABLE}" "${_DESCRIPTOR_SET_FILE}" "${_PROTO_FILE}" ${_DEPS}
    VERBATIM)

  cg_py_library(
    NAME "${_RULE_NAME}"
    GENERATED_SRCS ${_PY_GRPC_PROTO_FILES}
    )
endfunction()
