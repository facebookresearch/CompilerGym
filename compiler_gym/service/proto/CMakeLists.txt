# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME "proto"
  SRCS
    "__init__.py"
    "py_converters.py"
  PUBLIC
  DEPS
    "::compiler_gym_service_py"
    "::compiler_gym_service_py_grpc"
    compiler_gym::spaces::commandline
    compiler_gym::spaces::dict
    compiler_gym::spaces::discrete
    compiler_gym::spaces::named_discrete
    compiler_gym::spaces::scalar
    compiler_gym::spaces::tuple
)

proto_library(
  NAME
    compiler_gym_service
  SRCS
    compiler_gym_service.proto
  PUBLIC
)

py_proto_library(
    NAME
      compiler_gym_service_py
    DEPS
      ::compiler_gym_service
)

cc_proto_library(
  NAME
    compiler_gym_service_cc
  DEPS
    ::compiler_gym_service
  PUBLIC
)

cc_grpc_library(
    NAME compiler_gym_service_cc_grpc
    SRCS ::compiler_gym_service
    GRPC_ONLY
    PUBLIC
    DEPS ::compiler_gym_service_cc
)

py_grpc_library(
    NAME "compiler_gym_service_py_grpc"
    SRCS "::compiler_gym_service"
    DEPS "::compiler_gym_service_py"
)
