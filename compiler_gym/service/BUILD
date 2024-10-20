# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

py_library(
    name = "service",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":compilation_session",
        ":connection",
        # TODO(github.com/facebookresearch/CompilerGym/pull/633):
        # add this after circular dependencies are resolved
        # ":client_service_compiler_env",
        ":service_cache",
        "//compiler_gym/errors",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "compilation_session",
    srcs = ["compilation_session.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)

cc_library(
    name = "CompilationSession",
    srcs = ["CompilationSession.cc"],
    hdrs = ["CompilationSession.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":CompilerGymServiceContext",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "CompilerGymServiceContext",
    srcs = ["CompilerGymServiceContext.cc"],
    hdrs = ["CompilerGymServiceContext.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

py_library(
    name = "connection",
    srcs = ["connection.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":service_cache",
        "//compiler_gym/errors",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "client_service_compiler_env",
    srcs = ["client_service_compiler_env.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":compilation_session",
        ":connection",
        "//compiler_gym:compiler_env_state",
        "//compiler_gym:validation_result",
        "//compiler_gym/datasets",
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/errors",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
        "//compiler_gym/views",
    ],
)

py_library(
    name = "service_cache",
    srcs = ["service_cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/util",
    ],
)
