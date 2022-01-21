# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    runtime
  SRCS
    "__init__.py"
  DEPS
    ::create_and_run_compiler_gym_service
  PUBLIC
)

cg_cc_library(
  NAME
    cc_runtime
  HDRS
    "Runtime.h"
  DEPS
    ::CreateAndRunCompilerGymServiceImpl
  PUBLIC
)

cg_py_library(
  NAME
    benchmark_cache
  SRCS
    "benchmark_cache.py"
  DEPS
    compiler_gym::service::proto::proto
  PUBLIC
)

cg_cc_library(
  NAME
    BenchmarkCache
  HDRS
    "BenchmarkCache.h"
  SRCS
    "BenchmarkCache.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
  ABS_DEPS
    Boost::filesystem
    grpc++
    glog::glog
  PUBLIC
)

cg_py_library(
  NAME
    compiler_gym_service
  SRCS
    "compiler_gym_service.py"
  DEPS
    ::benchmark_cache
    compiler_gym::service::compilation_session
    compiler_gym::service::proto::proto
    compiler_gym::util::util
  PUBLIC
)

cg_cc_library(
  NAME
    CompilerGymService
  HDRS
    "CompilerGymService.h"
    "CompilerGymServiceImpl.h"
  DEPS
    ::BenchmarkCache
    ::CompilerGymServiceImpl
    compiler_gym::service::CompilationSession
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::service::proto::compiler_gym_service_cc_grpc
  ABS_DEPS
    Boost::filesystem
    grpc++
  PUBLIC
)

cg_cc_library(
  NAME
    CompilerGymServiceImpl
  HDRS
    "CompilerGymServiceImpl.h"
  DEPS
    compiler_gym::util::GrpcStatusMacros
    compiler_gym::util::Version
  ABS_DEPS
    fmt
    glog::glog
  PUBLIC
)

cg_py_library(
  NAME
    create_and_run_compiler_gym_service
  SRCS
    "create_and_run_compiler_gym_service.py"
  DEPS
    ::compiler_gym_service
    compiler_gym::service::proto::proto
    compiler_gym::util::util
  PUBLIC
)

cg_cc_library(
  NAME
    CreateAndRunCompilerGymServiceImpl
  HDRS
    "CreateAndRunCompilerGymServiceImpl.h"
  SRCS
    "CreateAndRunCompilerGymServiceImpl.cc"
  DEPS
    ::CompilerGymService
    compiler_gym::util::GrpcStatusMacros
  ABS_DEPS
    Boost::filesystem
    grpc++
    gflags
    glog::glog
  PUBLIC
)
