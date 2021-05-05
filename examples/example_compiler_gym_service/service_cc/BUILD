# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "compiler_gym-example-service-cc",
    srcs = ["RunService.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/core:cc_core",
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/util:GrpcStatusMacros",
        "@boost//:filesystem",
        "@fmt",
        "@glog",
    ],
)
