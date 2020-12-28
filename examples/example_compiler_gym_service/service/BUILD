# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cc_binary(
    name = "service",
    srcs = ["RunService.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":ExampleService",
        "//compiler_gym/util:RunService",
    ],
)

cc_library(
    name = "ExampleService",
    srcs = ["ExampleService.cc"],
    hdrs = ["ExampleService.h"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/service/proto:compiler_gym_service_cc_grpc",
        "//compiler_gym/util:GrpcStatusMacros",
        "//compiler_gym/util:Version",
        "@boost//:filesystem",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
