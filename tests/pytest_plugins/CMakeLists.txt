# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    gcc
  SRCS
    "gcc.py"
  DEPS
    ::common
    compiler_gym::envs::gcc::gcc
  TESTONLY
  PUBLIC
)

cg_py_library(
  NAME
    llvm
  SRCS
    "llvm.py"
  DATA
    compiler_gym::envs::llvm::service::passes::actions_genfiles
    compiler_gym::third_party::cbench::benchmarks_list
  DEPS
    compiler_gym::envs::llvm::llvm
    compiler_gym::third_party::llvm::llvm
    compiler_gym::util::util
  TESTONLY
  PUBLIC
)

cg_py_library(
  NAME
    common
  SRCS
    "common.py"
  DEPS
    compiler_gym::util::util
  TESTONLY
  PUBLIC
)

cg_py_library(
  NAME
    random_util
  SRCS
    "random_util.py"
  DEPS
    compiler_gym::compiler_gym
  TESTONLY
  PUBLIC
)
