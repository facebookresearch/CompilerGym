# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "loop_tool",
    srcs = [
        "__init__.py",
        "loop_tool_env.py",
    ],
    visibility = ["//visibility:public"],
    data = [
      ":loop_tool-service-py",
    ],
    deps = [
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime",
    ],
)

py_binary(
    name = "loop_tool-service-py",
    srcs = ["loop_tool_env.py"],
    main = "loop_tool_env.py",
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime",
    ],
)

