# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

py_binary(
    name = "loops-opt-service-py",
    srcs = ["loops_opt_service.py"],
    data = [
        ":compute_programl",
        "//compiler_gym/third_party/autophase",
        "//examples/loop_optimizations_service/opt_loops",
    ],
    main = "loops_opt_service.py",
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime",
        "//compiler_gym/spaces",
        "//compiler_gym/third_party/inst2vec",
        "//compiler_gym/third_party/llvm",
    ],
)

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
        "@programl//programl/proto:programl_cc",
    ],
)
