# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    envs
  SRCS
    "__init__.py"
  DEPS
    ::compiler_env
    compiler_gym::envs::gcc::gcc
    compiler_gym::envs::llvm::llvm
    compiler_gym::envs::loop_tool::loop_tool
  PUBLIC
)

cg_py_library(
  NAME
    compiler_env
  SRCS
    "compiler_env.py"
  DEPS
    compiler_gym::compiler_env_state
    compiler_gym::datasets::datasets
    compiler_gym::service::service
    compiler_gym::service::proto::proto
    compiler_gym::spaces::spaces
    compiler_gym::util::util
    compiler_gym::validation_result
    compiler_gym::views::views
  PUBLIC
)
