# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    cgra
  SRCS
    "__init__.py"
    "cgra_rewards.py"
    "Operations.py"
    "compile_settings.py"
    "DFG.py"
  DATA
    compiler_gym::envs::cgra::service::service
  DEPS
    compiler_gym::service::client_service_compiler_env
    compiler_gym::envs::cgra::datasets::datasets
    compiler_gym::engs::cgra::architectures::architectures
    compiler_gym::errors::errors
    compiler_gym::service::runtime::runtime
    compiler_gym::util::util
  PUBLIC
)
