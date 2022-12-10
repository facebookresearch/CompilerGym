# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

set(ENVS_DEPS
    ::compiler_env
    compiler_gym::envs::gcc::gcc
    compiler_gym::envs::cgra::cgra
    compiler_gym::envs::loop_tool::loop_tool
)
if(COMPILER_GYM_ENABLE_LLVM_ENV)
    list(APPEND ENVS_DEPS compiler_gym::envs::llvm::llvm)
endif()
if(COMPILER_GYM_ENABLE_MLIR_ENV)
    list(APPEND ENVS_DEPS compiler_gym::envs::mlir::mlir)
endif()
cg_py_library(
  NAME
    envs
  SRCS
    "__init__.py"
  DEPS ${ENVS_DEPS}
  PUBLIC
)

cg_py_library(
  NAME
    compiler_env
  SRCS
    "compiler_env.py"
  DEPS
    compiler_gym::spaces::spaces
    compiler_gym::util::util
    compiler_gym::views::views
    compiler_gym::compiler_env_state
    compiler_gym::validation_result
    compiler_gym::datasets::datasets
  PUBLIC
)
