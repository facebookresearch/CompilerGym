# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    outs = ["clang"],
    cmd = "cp $< $@",
    visibility = ["//visibility:public"],
)
