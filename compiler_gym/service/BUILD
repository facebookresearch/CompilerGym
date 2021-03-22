# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

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
    visibility = ["//visibility:public"],
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
