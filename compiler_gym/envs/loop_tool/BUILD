# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "loop_tool",
    srcs = [
        "__init__.py",
        "loop_tool_env.py",
    ],
    data = ["//compiler_gym/envs/loop_tool/service"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service:client_service_compiler_env",
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime",
    ],
)
