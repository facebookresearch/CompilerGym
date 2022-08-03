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
        "llvm_lexer_token_info.cc"
    ],
    hdrs = [
        "escape.h",
        "LLLexer.h",
        "LLToken.h",
        "llvm_lexer_token_info.h",
        "llvm_lexer.h",
        "lexed_ir.h",
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