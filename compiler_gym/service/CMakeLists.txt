# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    service
  SRCS
    "__init__.py"
  DEPS
    ::compilation_session
    ::connection
    compiler_gym::service::proto::proto
  PUBLIC
)

cg_py_library(
  NAME
    compilation_session
  SRCS
    "compilation_session.py"
  DEPS
    compiler_gym::service::proto::proto
  PUBLIC
)

cg_cc_library(
  NAME
    CompilationSession
  HDRS
    "CompilationSession.h"
  SRCS
    "CompilationSession.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
  ABS_DEPS
    Boost::filesystem
    grpc++
  PUBLIC
)

cg_py_library(
  NAME
    connection
  SRCS
    "connection.py"
  DEPS
    compiler_gym::service::proto::proto
    compiler_gym::util::util
  PUBLIC
)
