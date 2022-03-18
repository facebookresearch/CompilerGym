# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "compute_programl",
    srcs = ["compute_programl.cc"],
    copts = [
        "-DGOOGLE_PROTOBUF_NO_RTTI",
        "-fno-rtti",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@glog",
        "@llvm//10.0.0",
        "@nlohmann_json//:json",
        "@programl//programl/graph/format:node_link_graph",
        "@programl//programl/ir/llvm:llvm-10",
    ],
)
