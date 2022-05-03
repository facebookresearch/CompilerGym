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
    srcs = [
        "__init__.py",
        "py_converters.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_gym_service_py",
        ":compiler_gym_service_py_grpc",
        "//compiler_gym/spaces:box",
        "//compiler_gym/spaces:commandline",
        "//compiler_gym/spaces:dict",
        "//compiler_gym/spaces:discrete",
        "//compiler_gym/spaces:named_discrete",
        "//compiler_gym/spaces:permutation",
        "//compiler_gym/spaces:scalar",
        "//compiler_gym/spaces:sequence",
        "//compiler_gym/spaces:space_sequence",
        "//compiler_gym/spaces:tuple",
    ],
)

proto_library(
    name = "compiler_gym_service",
    srcs = ["compiler_gym_service.proto"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:any_proto"],
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

cc_library(
    name = "proto_cc",
    srcs = ["Proto.cc"],
    hdrs = ["Proto.h"],
    visibility = ["//visibility:public"],
    deps = [
        "compiler_gym_service_cc",
        "@glog",
        "@magic_enum",
    ],
)
