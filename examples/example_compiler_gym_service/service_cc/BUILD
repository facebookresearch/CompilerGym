# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "compiler_gym-example-service-cc",
    srcs = ["ExampleService.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service:CompilationSession",
        "//compiler_gym/service/runtime:cc_runtime",
        "//compiler_gym/util:Unreachable",
        "@com_github_grpc_grpc//:grpc++",
        "@fmt",
    ],
)
