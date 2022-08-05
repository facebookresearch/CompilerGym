# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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
