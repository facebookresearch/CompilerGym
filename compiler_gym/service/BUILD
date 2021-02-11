# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")

py_library(
    name = "service",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":connection",
        ":proto2py",
        "//compiler_gym/service/proto",
    ],
)

py_library(
    name = "connection",
    srcs = ["connection.py"],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "proto2py",
    srcs = ["proto2py.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)
