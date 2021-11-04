# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CMakeParseArguments)
include(cmake_macros)
include(cmake_py_library)

function(proto_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC"
    "NAME;SRCS"
    "DEPS"
    ${ARGN}
  )

  rename_bazel_targets("${_RULE_DEPS}" _RULE_DEPS)
  rename_bazel_targets("${_RULE_NAME}" _RULE_NAME)

  set(_SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}")
  set(_DST_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SRCS}.bin")
  get_filename_component(_DST_DIR "${_DST_FILE}" DIRECTORY)

  add_custom_command(
    OUTPUT "${_DST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --descriptor_set_out "${_DST_FILE}"
      "${_SRC_FILE}"
    DEPENDS "${_SRC_FILE}" ${_RULE_DEPS})

  add_custom_target(${_RULE_NAME} ALL DEPENDS "${_DST_FILE}")
  set_target_properties(${_RULE_NAME} PROPERTIES PROTO_BYPRODUCTS "${_DST_FILE}")
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

  rename_bazel_targets("${_RULE_DEPS}" _RULE_DEPS)
  rename_bazel_targets("${_RULE_NAME}" _RULE_NAME)

  get_target_as_relative_dir(${_RULE_NAME} _HEADER_DST_DIR)
  set(_HEADER_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_DST_DIR}")
  get_target_property(_SRC_FILE ${_RULE_DEPS} PROTO_BYPRODUCTS)
  get_filename_component(_SRC_FILENAME "${_SRC_FILE}" NAME)
  get_cc_proto_out_files("${_SRC_FILENAME}" _CC_PROTO_FILES)
  list(TRANSFORM _CC_PROTO_FILES PREPEND "${_HEADER_DST_DIR}/")

  add_custom_command(
    OUTPUT ${_CC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_HEADER_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --descriptor_set_in="${_SRC_FILE}"
      --cpp_out "${_HEADER_DST_DIR}"
    DEPENDS "${_SRC_FILE}" ${_RULE_DEPS})

  add_library(${_RULE_NAME} ${_CC_PROTO_FILES})
  target_link_libraries(${_RULE_NAME} PRIVATE protobuf::libprotobuf)
  target_include_directories(${_RULE_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
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

  rename_bazel_targets("${_RULE_DEPS}" _DEPS)

  get_target_property(_SRC_FILE ${_DEPS} PROTO_BYPRODUCTS)
  get_filename_component(_SRC_FILENAME "${_SRC_FILE}" NAME)
  get_py_proto_out_files("${_SRC_FILENAME}" _PY_PROTO_FILES)
  set(_PYTHON_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(_ABS_PATH_PY_PROTO_FILES ${_PY_PROTO_FILES})
  list(TRANSFORM _ABS_PATH_PY_PROTO_FILES PREPEND "${_PYTHON_DST_DIR}/")

  add_custom_command(
    OUTPUT ${_ABS_PATH_PY_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_PYTHON_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --descriptor_set_in="${_SRC_FILE}"
      --python_out "${_PYTHON_DST_DIR}"
    DEPENDS "${_SRC_FILE}" ${_DEPS})

  cmake_py_library(
    NAME "${_RULE_NAME}"
    GENERATED_SRCS ${_PY_PROTO_FILES}
    )
endfunction()
