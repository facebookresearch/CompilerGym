# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#
# This package exposes the LLVM optimization pipeline as a CompilerGym service.
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "opt_loops",
    srcs = [
        "opt_loops.cc",
    ],
    copts = [
        "-Wall",
        "-fdiagnostics-color=always",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":IRCanonicalizer",
        "@llvm//10.0.0",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "IRCanonicalizer",
    srcs = [
        "IRCanonicalizer.cc",
    ],
    hdrs = [
        "IRCanonicalizer.h",
    ],
    copts = [
        "-Wall",
        "-fdiagnostics-color=always",
        "-fno-rtti",
    ],
    deps = [
        "@llvm//10.0.0",
    ],
)
