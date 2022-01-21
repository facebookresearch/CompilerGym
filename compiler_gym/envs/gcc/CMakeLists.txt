# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    gcc
  SRCS
    "__init__.py"
    "gcc.py"
    "gcc_env.py"
    "gcc_rewards.py"
  DATA
    compiler_gym::envs::gcc::service::service
  DEPS
    compiler_gym::envs::compiler_env
    compiler_gym::envs::gcc::datasets::datasets
    compiler_gym::service::service
    compiler_gym::service::runtime::runtime
    compiler_gym::util::util
  PUBLIC
)
