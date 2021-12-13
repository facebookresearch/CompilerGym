# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    autophase
  SRCS
    "__init__.py"
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS analysis core support)
cg_cc_library(
  NAME
    InstCount
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  HDRS
    "InstCount.h"
  SRCS
    "InstCount.cc"
  ABS_DEPS
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support)
cg_cc_binary(
  NAME
    compute_autophase
  SRCS
    "compute_autophase.cc"
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  DEPS
    ::InstCount
  ABS_DEPS
    glog::glog
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)

cg_genrule(
  NAME libLLVMPolly
  OUTS "libLLVMPolly.so"
  COMMAND
    "cp $<TARGET_FILE:LLVMPolly> $@"
  ABS_DEPENDS
    LLVMPolly
)
