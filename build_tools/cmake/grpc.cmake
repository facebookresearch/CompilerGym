# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)
include(CMakeParseArguments)
include(cmake_macros)
include(cmake_py_library)
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

  rename_bazel_targets("${_RULE_DEPS}" _RULE_DEPS)
  rename_bazel_targets("${_RULE_NAME}" _RULE_NAME)
  rename_bazel_targets("${_RULE_SRCS}" _RULE_SRCS)

  get_target_as_relative_dir(${_RULE_NAME} _HEADER_DST_DIR)
  set(_HEADER_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/${_HEADER_DST_DIR}")
  get_target_property(_SRC_FILE ${_RULE_SRCS} PROTO_BYPRODUCTS)
  get_filename_component(_SRC_FILENAME "${_SRC_FILE}" NAME)
  get_cc_grpc_proto_out_files("${_SRC_FILENAME}" _GRPC_PROTO_FILES)
  list(TRANSFORM _CC_PROTO_FILES PREPEND "${_HEADER_DST_DIR}/")

  add_custom_command(
    OUTPUT ${_GRPC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_HEADER_DST_DIR}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
      --descriptor_set_in="${_SRC_FILE}"
      --grpc_out "${_HEADER_DST_DIR}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    DEPENDS "${_SRC_FILE}" ${_RULE_DEPS})

  add_library(${_RULE_NAME} ${_GRPC_PROTO_FILES})
  target_link_libraries(${_RULE_NAME} PRIVATE com_github_grpc_grpc::grpc++ ${_RULE_DEPS})
  target_include_directories(${_RULE_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
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

  rename_bazel_targets("${_RULE_DEPS}" _DEPS)
  rename_bazel_targets("${_RULE_SRCS}" _SRCS)

  get_target_property(_SRC_FILE ${_SRCS} PROTO_BYPRODUCTS)
  get_filename_component(_SRC_FILENAME "${_SRC_FILE}" NAME)
  get_py_grpc_proto_out_files("${_SRC_FILENAME}" _PY_GRPC_PROTO_FILES)
  set(_PYTHON_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(_ABS_PATH_PY_GRPC_PROTO_FILES ${_PY_GRPC_PROTO_FILES})
  list(TRANSFORM _ABS_PATH_PY_GRPC_PROTO_FILES PREPEND "${_PYTHON_DST_DIR}/")

  add_custom_command(
    OUTPUT ${_ABS_PATH_PY_GRPC_PROTO_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_PYTHON_DST_DIR}"
    COMMAND "${Python_EXECUTABLE}"
      -m grpc_tools.protoc
      --descriptor_set_in="${_SRC_FILE}"
      ---grpc_python_out="${_PYTHON_DST_DIR}"
    DEPENDS "${_SRC_FILE}" ${_DEPS})

  cmake_py_library(
    NAME "${_RULE_NAME}"
    GENERATED_SRCS ${_PY_GRPC_PROTO_FILES}
    )
endfunction()
