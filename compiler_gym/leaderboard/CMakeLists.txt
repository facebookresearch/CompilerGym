# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    leaderboard
  SRCS
    "__init__.py"
  DEPS
    ::llvm_instcount
  PUBLIC
)

cg_py_library(
  NAME
    llvm_instcount
  SRCS
    "llvm_instcount.py"
  DEPS
    compiler_gym::bin::validate
    compiler_gym::compiler_env_state
    compiler_gym::envs::envs
    compiler_gym::util::util
  PUBLIC
)
