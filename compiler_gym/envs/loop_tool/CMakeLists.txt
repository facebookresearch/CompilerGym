# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME loop_tool
  SRCS
    "__init__.py"
    "loop_tool_env.py"
  DATA compiler_gym::envs::loop_tool::service::service
  DEPS
    compiler_gym::envs::compiler_env
    compiler_gym::service::service
    compiler_gym::service::proto::proto
    compiler_gym::service::runtime::runtime
  PUBLIC
)
