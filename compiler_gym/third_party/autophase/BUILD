# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Autophase https://github.com/ucb-bar/autophase
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_library")

py_library(
    name = "autophase",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "InstCount",
    srcs = ["InstCount.cc"],
    hdrs = ["InstCount.h"],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//10.0.0",
    ],
)
