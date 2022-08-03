# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

llvm_map_components_to_libnames(_LLVM_LIBS
  core support bitwriter
)
cg_cc_library(
    NAME LLVMLexer
    SRCS
      LLLexer.cc
      llvm_lexer_token_info.cc
    HDRS
      escape.h
      LLLexer.h
      LLToken.h
      llvm_lexer_token_info.h
      llvm_lexer.h
      LexedIR.h
    ABS_DEPS
      ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)
