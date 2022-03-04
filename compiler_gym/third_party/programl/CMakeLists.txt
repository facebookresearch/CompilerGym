# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support)
cg_cc_binary(
  NAME
    compute_programl
  SRCS
    "compute_programl.cc"
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  DEPS
    nlohmann_json::nlohmann_json
    ProGraML::graph::format::node_link_graph
    ProGraML::ir::llvm::llvm-10
    ProGraML::proto::programl_cc
  ABS_DEPS
    glog::glog
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)
