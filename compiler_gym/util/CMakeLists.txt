# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_library(
  NAME
    util
  SRCS
    "__init__.py"
    "capture_output.py"
    "commands.py"
    "debug_util.py"
    "decorators.py"
    "download.py"
    "executor.py"
    "filesystem.py"
    "gym_type_hints.py"
    "logging.py"
    "logs.py"
    "minimize_trajectory.py"
    "parallelization.py"
    "registration.py"
    "runfiles_path.py"
    "shell_format.py"
    "statistics.py"
    "tabulate.py"
    "temporary_working_directory.py"
    "thread_pool.py"
    "timer.py"
    "truncate.py"
  GENERATED_SRCS
    "$<TARGET_PROPERTY:compiler_gym__util__make_version,LOCATION>"
  DEPS
    make_version
    # TODO(boian): verify if needed
    #unconverted_name:@rules_python//python/runfiles
  PUBLIC
)

cg_cc_library(
  NAME
    EnumUtil
  SRCS
    "EnumUtil.h"
  ABS_DEPS
    grpc++
    fmt
    magic_enum
  PUBLIC
)

cg_cc_library(
  NAME
    GrpcStatusMacros
  HDRS
    "GrpcStatusMacros.h"
  ABS_DEPS
    Boost::headers
    grpc++
    fmt
    glog::glog
  PUBLIC
)

cg_cc_library(
  NAME
    RunfilesPath
  HDRS
    "RunfilesPath.h"
  SRCS
    "RunfilesPath.cc"
  ABS_DEPS
    Boost::filesystem
    fmt
  PUBLIC
)

cg_cc_library(
  NAME
    StrLenConstexpr
  HDRS
    "StrLenConstexpr.h"
  PUBLIC
)

cg_cc_library(
  NAME
    Subprocess
  HDRS
    "Subprocess.h"
  SRCS
    "Subprocess.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
  ABS_DEPS
    Boost::filesystem
    Boost::headers
    grpc++
    fmt
  PUBLIC
)

cg_cc_library(
  NAME
    Unreachable
  HDRS
    "Unreachable.h"
  ABS_DEPS
    glog::glog
  PUBLIC
)

cg_genrule(
  NAME make_version
  OUTS "version.py"
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION"
  COMMAND "echo \"__version__ = \\\"$(cat \"${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION\")\\\"\" > $@"
)

cg_cc_library(
  NAME
    Version
  HDRS
    "$<TARGET_PROPERTY:compiler_gym__util__make_version_header,BINARY_DIR>/Version.h"
  PUBLIC
)

cg_genrule(
  NAME make_version_header
  OUTS "Version.h"
  SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION"
  COMMAND "echo \"#define COMPILER_GYM_VERSION \\\"$(cat \"${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION\")\\\"\" > $@"
)
