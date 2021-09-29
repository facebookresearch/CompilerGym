# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    spaces
  SRCS
    "__init__.py"
  DEPS
    ::box
    ::commandline
    ::dict
    ::discrete
    ::named_discrete
    ::reward
    ::scalar
    ::sequence
    ::tuple
  PUBLIC
)

cg_py_library(
  NAME box
  SRCS box.py
)

cg_py_library(
  NAME
    commandline
  SRCS
    "commandline.py"
  DEPS
    ::named_discrete
  PUBLIC
)

cg_py_library(
  NAME dict
  SRCS dict.py
)

cg_py_library(
  NAME discrete
  SRCS discrete.py
)

cg_py_library(
  NAME
    named_discrete
  SRCS
    "named_discrete.py"
  DEPS
    ::discrete
  PUBLIC
)

cg_py_library(
  NAME
    reward
  SRCS
    "reward.py"
  DEPS
    ::scalar
    compiler_gym::service::service
    compiler_gym::util::util
  PUBLIC
)

cg_py_library(
  NAME
    scalar
  SRCS
    "scalar.py"
  PUBLIC
)

cg_py_library(
  NAME
    sequence
  SRCS
    "sequence.py"
  DEPS
    ::scalar
  PUBLIC
)

cg_py_library(
  NAME tuple
  SRCS "tuple.py"
)
