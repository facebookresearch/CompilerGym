# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if(NOT COMPILER_GYM_ENABLE_LLVM_ENV)
    return()
endif()

cg_add_all_subdirs()

llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support passes)
cg_cc_library(
  NAME
     IRCanonicalizer
  SRCS
    "IRCanonicalizer.cc"
  COPTS
    "-fno-rtti"
  ABS_DEPS
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)
