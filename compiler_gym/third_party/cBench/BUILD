# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

filegroup(
    name = "benchmarks_list",
    srcs = ["benchmarks.txt"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "make_llvm_module",
    srcs = ["make_llvm_module.py"],
    data = [
        "@llvm//10.0.0:clang",
        "@llvm//10.0.0:llvm-link",
    ],
    deps = [
        "//compiler_gym/util",
    ],
)

genrule(
    name = "crc32",
    srcs = [
        "@cBench//:all",
        "@cBench//:readme",
    ],
    outs = ["cBench/crc32.bc"],
    cmd = "mkdir -p $(@D) && $(location :make_llvm_module) --path=$$(dirname $(location @cBench//:readme))/telecom_CRC32 --output_path=$@",
    tools = [":make_llvm_module"],
    visibility = ["//visibility:public"],
)
