# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#
# This package exposes the LLVM optimization pipeline as a CompilerGym service.
load("@rules_cc//cc:defs.bzl", "cc_binary")

genrule(
    name = "libLLVMRuntimeDyld",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["libLLVMRuntimeDyld.a"],
    cmd = "cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/libLLVMRuntimeDyld.a $@",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "loop_unroller",
    srcs = [
        "loop_unroller.cc",
        #"@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:lib/libLLVMRuntimeDyld.a",
        ":libLLVMRuntimeDyld",
    ],
    copts = [
        "-Wall",
        "-fdiagnostics-color=always",
        "-fno-rtti",
        "-force_load",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//10.0.0",
        #":libLLVMRuntimeDyld",
    ],
)
