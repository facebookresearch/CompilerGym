# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

py_library(
    name = "service",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compilation_session",
        ":connection",
        "//compiler_gym/service/proto",
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
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)
