# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Autophase https://github.com/ucb-bar/autophase
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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

filegroup(
    name = "compute_autophase-files",
    srcs = [
        ":compute_autophase",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            ":libLLVMPolly",
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "compute_autophase-prelinked",
    srcs = ["compute_autophase.cc"],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":InstCount",
        "@glog",
        "@llvm//10.0.0",
    ],
)

genrule(
    name = "compute_autophase-bin",
    srcs = [":compute_autophase-prelinked"],
    outs = ["compute_autophase"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :compute_autophase-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :compute_autophase-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
)

genrule(
    name = "libLLVMPolly",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["libLLVMPolly.so"],
    cmd = "cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/LLVMPolly.so $@",
    visibility = ["//visibility:public"],
)
