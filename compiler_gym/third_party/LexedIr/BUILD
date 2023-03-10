# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

filegroup(
    name = "unlex-files",
    srcs = [
        ":unlex",
    ] + select({
        "@llvm//:darwin": [],
        "//conditions:default": [
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "unlex-prelinked",
    srcs = [
        "LLLexer.cc",
        "llvm_lexer_token_info.cc",
        "unlex.cc",
    ],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    deps = [
        "LexedIr",
        "@glog",
        "@llvm//10.0.0",
    ],
)

genrule(
    name = "unlex-bin",
    srcs = [":unlex-prelinked"],
    outs = ["unlex"],
    cmd = select({
        "@llvm//:darwin": (
            "cp $(location :unlex-prelinked) $@"
        ),
        "//conditions:default": (
            "cp $(location :unlex-prelinked) $@ && " +
            "chmod 666 $@ && " +
            "patchelf --set-rpath '$$ORIGIN' $@ && " +
            "chmod 555 $@"
        ),
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "LexedIr",
    srcs = [
        "LLLexer.cc",
        "llvm_lexer_token_info.cc",
    ],
    hdrs = [
        "LLLexer.h",
        "LLToken.h",
        "escape.h",
        "lexed_ir.h",
        "llvm_lexer.h",
        "llvm_lexer_token_info.h",
    ],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm//10.0.0",
    ],
)
