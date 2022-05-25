# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

if(NOT COMPILER_GYM_ENABLE_LLVM_ENV)
    return()
endif()

llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support passes)
cg_cc_binary(
  NAME
    compute_programl
  SRCS
    "compute_programl.cc"
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  ABS_DEPS
    glog::glog
    ${_LLVM_LIBS}
    ProGraML::graph::format::node_link_graph
    ProGraML::ir::llvm::llvm-10
    ProGraML::proto::programl_cc
    nlohmann_json::nlohmann_json
    protobuf::libprotobuf
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

add_custom_target(
    link_compute_programl_target
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compiler_gym/third_party/programl/compute_programl
        ${CMAKE_SOURCE_DIR}/compiler_gym/third_party/programl/compute_programl
)
