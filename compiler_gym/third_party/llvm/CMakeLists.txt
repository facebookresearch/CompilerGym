# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    llvm
  SRCS
    "__init__.py"
  DEPS
    compiler_gym::util::util
  PUBLIC
)

llvm_map_components_to_libnames(llvm_libs support irreader core)
cg_cc_binary(
  NAME
    compute_ir_instruction_count
  SRCS
    "compute_ir_instruction_count.cc"
  COPTS
    "-DGOOGLE_PROTOBUF_NO_RTTI"
    "-fno-rtti"
  ABS_DEPS
    glog::glog
    ${llvm_libs}
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

cg_py_library(
  NAME
    instcount
  SRCS
    "instcount.py"
  PUBLIC
)

llvm_map_components_to_libnames(llvm_libs support core analysis)
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
    glog::glog
    ${llvm_libs}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
  PUBLIC
)

cg_cc_binary(
  NAME
    PrintInstCountFeatureNames
  SRCS
    "PrintInstCountFeatureNames.cc"
  DEPS
    ::InstCount
)
