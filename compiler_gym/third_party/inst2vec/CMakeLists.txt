# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_genrule(
  NAME dictionary
  OUTS "dictionary.pickle"
  SRCS "dictionary.tar.bz2"
  COMMAND "tar xjf ${CMAKE_CURRENT_SOURCE_DIR}/dictionary.tar.bz2 -C $(@D)"
)

cg_genrule(
  NAME embeddings
  OUTS "embeddings.pickle"
  SRCS "embeddings.tar.bz2"
  COMMAND "tar xjf ${CMAKE_CURRENT_SOURCE_DIR}/embeddings.tar.bz2 -C $(@D)"
)

cg_py_library(
  NAME
    inst2vec
  SRCS
    "__init__.py"
  DATA
    ::dictionary
    ::embeddings
  DEPS
    ::inst2vec_preprocess
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    inst2vec_preprocess
  SRCS
    "inst2vec_preprocess.py"
  DEPS
    ::rgx_utils
  PUBLIC
)

cg_py_library(
  NAME
    rgx_utils
  SRCS
    "rgx_utils.py"
  PUBLIC
)
