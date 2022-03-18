# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if(NOT COMPILER_GYM_ENABLE_MLIR_ENV)
    return()
endif()

cg_add_all_subdirs()

cg_py_library(
  NAME
    mlir
  SRCS
    "__init__.py"
  DATA
    compiler_gym::envs::mlir::service::service
  DEPS
    ::mlir_env
  PUBLIC
)

cg_py_library(
  NAME
    mlir_env
  SRCS
    "mlir_env.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::service::client_service_compiler_env
    compiler_gym::spaces::spaces
  PUBLIC
)
