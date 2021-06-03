# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This package implements the CompilerGym service runtime, which is the utility
# code that creates RPC servers and dispatches to CompilationServices.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "runtime",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":create_and_run_compiler_gym_service",
    ],
)

cc_library(
    name = "cc_runtime",
    hdrs = ["Runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":CreateAndRunCompilerGymServiceImpl",
    ],
)

py_library(
    name = "benchmark_cache",
    srcs = ["benchmark_cache.py"],
    visibility = ["//tests/service/runtime:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)

cc_library(
    name = "BenchmarkCache",
    srcs = ["BenchmarkCache.cc"],
    hdrs = ["BenchmarkCache.h"],
    visibility = ["//tests/service/runtime:__subpackages__"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@glog",
    ],
)

py_library(
    name = "compiler_gym_service",
    srcs = ["compiler_gym_service.py"],
    deps = [
        ":benchmark_cache",
        "//compiler_gym/service:compilation_session",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

cc_library(
    name = "CompilerGymService",
    hdrs = [
        "CompilerGymService.h",
        "CompilerGymServiceImpl.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":BenchmarkCache",
        ":CompilerGymServiceImpl",
        "//compiler_gym/service:CompilationSession",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "CompilerGymServiceImpl",
    hdrs = ["CompilerGymServiceImpl.h"],
    deps = [
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:Version",
        "@fmt",
        "@glog",
    ],
)

py_library(
    name = "create_and_run_compiler_gym_service",
    srcs = ["create_and_run_compiler_gym_service.py"],
    deps = [
        ":compiler_gym_service",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

cc_library(
    name = "CreateAndRunCompilerGymServiceImpl",
    srcs = ["CreateAndRunCompilerGymServiceImpl.cc"],
    hdrs = ["CreateAndRunCompilerGymServiceImpl.h"],
    deps = [
        ":CompilerGymService",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
        "@gflags",
        "@glog",
    ],
)
