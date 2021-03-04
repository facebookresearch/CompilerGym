# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

filegroup(
    name = "clang",
    srcs = [
        "bin/clang",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
            "lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a",
            "lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a",
            "lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a",
            "lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a",
        ],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "llvm-diff",
    srcs = ["bin/llvm-diff"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "llvm-link",
    srcs = ["bin/llvm-link"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "llvm-size",
    srcs = ["bin/llvm-size"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "opt",
    srcs = ["bin/opt"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "llvm-stress",
    srcs = ["bin/llvm-stress"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "make_clang",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:clang",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
        ],
    }),
    outs = ["bin/clang"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_llvm-diff",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-diff",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-diff",
        ],
    }),
    outs = ["bin/llvm-diff"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_llvm-link",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-link",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-link",
        ],
    }),
    outs = ["bin/llvm-link"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_llvm-size",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-size",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-size",
        ],
    }),
    outs = ["bin/llvm-size"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_opt",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:opt",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:opt",
        ],
    }),
    outs = ["bin/opt"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_llvm-stress",
    srcs = select({
        "@llvm//:darwin": [
            "@clang-llvm-10.0.0-x86_64-apple-darwin//:llvm-stress",
        ],
        "//conditions:default": [
            "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:llvm-stress",
        ],
    }),
    outs = ["bin/llvm-stress"],
    cmd = "mkdir -p $$(dirname $@) && cp $< $@",
)

genrule(
    name = "make_libclang_rt.asan-x86_64.a",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a"],
    cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.asan-x86_64.a $@",
)

genrule(
    name = "make_libclang_rt.tsan-x86_64.a",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a"],
    cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.tsan-x86_64.a $@",
)

genrule(
    name = "make_libclang_rt.msan-x86_64.a",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a"],
    cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.msan-x86_64.a $@",
)

genrule(
    name = "make_libclang_rt.ubsan_standalone-x86_64.a",
    srcs = [
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:all_files",
        "@clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang",
    ],
    outs = ["lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a"],
    cmd = "mkdir -p $$(dirname $@) && cp $$(dirname $(location @clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04//:clang))/../lib/clang/10.0.0/lib/linux/libclang_rt.ubsan_standalone-x86_64.a $@",
)
