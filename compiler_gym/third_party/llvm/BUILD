# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "llvm",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/util",
    ],
)

py_library(
    name = "instcount",
    srcs = ["instcount.py"],
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
        "@glog",
        "@llvm//10.0.0",
    ],
)

cc_binary(
    name = "PrintInstCountFeatureNames",
    srcs = ["PrintInstCountFeatureNames.cc"],
    deps = [":InstCount"],
)

# The following targets define the rules for how to assemble and compress the
# tarball of LLVM data that is downloaded at runtime by the ":llvm" library.
# It is commented out to prevent globs like "bazel test //..." from running
# them, as they are only needed for updating the distributed LLVM archives.
#
# genrule(
#     name = "llvm_data_tar",
#     srcs = [":llvm_data"],
#     outs = ["llvm-v0.tar.bz2"],
#     cmd = "mkdir -p $(@D)/lib && tar cjfh $@ -C $(@D) LICENSE bin lib",
# )
#
# filegroup(
#     name = "llvm_data",
#     srcs = [
#         "bin/clang",
#         "bin/llc",
#         "bin/lli",
#         "bin/llvm-as",
#         "bin/llvm-bcanalyzer",
#         "bin/llvm-config",
#         "bin/llvm-diff",
#         "bin/llvm-dis",
#         "bin/llvm-link",
#         "bin/llvm-mca",
#         "bin/llvm-size",
#         "bin/llvm-stress",
#         "bin/opt",
#         "LICENSE",
#     ] + select({
#         "@llvm//:darwin": [],
#         "//conditions:default": [
#             "lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a",
#             "lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a",
#             "lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a",
#             "lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a",
#         ],
#     }),
# )
#
# genrule(
#     name = "make_license",
#     srcs = ["LICENSE.txt"],
#     outs = ["LICENSE"],
#     cmd = "cat $< > $@",
# )
#
# genrule(
#     name = "make_clang",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:clang",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
#         ],
#     }),
#     outs = ["bin/clang"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_lli",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:lli",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:lli",
#         ],
#     }),
#     outs = ["bin/lli"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )

# genrule(
#     name = "make_llc",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llc",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llc",
#         ],
#     }),
#     outs = ["bin/llc"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-as",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-as",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-as",
#         ],
#     }),
#     outs = ["bin/llvm-as"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-mca",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-mca",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-mca",
#         ],
#     }),
#     outs = ["bin/llvm-mca"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-bcanalyzer",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-bcanalyzer",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-bcanalyzer",
#         ],
#     }),
#     outs = ["bin/llvm-bcanalyzer"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )

# genrule(
#     name = "make_llvm-config",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-config",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-config",
#         ],
#     }),
#     outs = ["bin/llvm-config"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-dis",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-dis",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-dis",
#         ],
#     }),
#     outs = ["bin/llvm-dis"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-diff",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-diff",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-diff",
#         ],
#     }),
#     outs = ["bin/llvm-diff"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-link",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-link",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-link",
#         ],
#     }),
#     outs = ["bin/llvm-link"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-size",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-size",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-size",
#         ],
#     }),
#     outs = ["bin/llvm-size"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_opt",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:opt",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:opt",
#         ],
#     }),
#     outs = ["bin/opt"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_llvm-stress",
#     srcs = select({
#         "@llvm//:darwin": [
#             "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-stress",
#         ],
#         "//conditions:default": [
#             "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-stress",
#         ],
#     }),
#     outs = ["bin/llvm-stress"],
#     cmd = "mkdir -p $$(dirname $@) && cp $< $@",
# )
#
# genrule(
#     name = "make_libclang_rt.asan-x86_64.a",
#     srcs = [
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
#     ],
#     outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a"],
#     cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a $@",
# )
#
# genrule(
#     name = "make_libclang_rt.tsan-x86_64.a",
#     srcs = [
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
#     ],
#     outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a"],
#     cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a $@",
# )
#
# genrule(
#     name = "make_libclang_rt.msan-x86_64.a",
#     srcs = [
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
#     ],
#     outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a"],
#     cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a $@",
# )
#
# genrule(
#     name = "make_libclang_rt.ubsan_standalone-x86_64.a",
#     srcs = [
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
#         "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
#     ],
#     outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a"],
#     cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a $@",
# )
