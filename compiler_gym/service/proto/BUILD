# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_gym_service_py",
        ":compiler_gym_service_py_grpc",
    ],
)

proto_library(
    name = "compiler_gym_service",
    srcs = ["compiler_gym_service.proto"],
    visibility = ["//visibility:public"],
)

py_proto_library(
    name = "compiler_gym_service_py",
    deps = [":compiler_gym_service"],
)

py_grpc_library(
    name = "compiler_gym_service_py_grpc",
    srcs = [":compiler_gym_service"],
    deps = [":compiler_gym_service_py"],
)

cc_proto_library(
    name = "compiler_gym_service_cc",
    visibility = ["//visibility:public"],
    deps = [":compiler_gym_service"],
)

cc_grpc_library(
    name = "compiler_gym_service_cc_grpc",
    srcs = [":compiler_gym_service"],
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":compiler_gym_service_cc"],
)
