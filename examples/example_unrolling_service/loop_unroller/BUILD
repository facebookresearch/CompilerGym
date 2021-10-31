# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#
# This package exposes the LLVM optimization pipeline as a CompilerGym service.
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "loop_unroller",
    srcs = ["loop_unroller.cc"],
    copts = [
        "-Wall",
        "-fdiagnostics-color=always",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//10.0.0",
    ],
)
