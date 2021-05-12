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
        ":benchmark_cache",
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
