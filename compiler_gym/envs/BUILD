# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "envs",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compiler_env",
        "//compiler_gym/envs/llvm",
    ],
)

py_library(
    name = "compiler_env",
    srcs = ["compiler_env.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        "//compiler_gym:compiler_env_state",
        "//compiler_gym:validation_result",
        "//compiler_gym/datasets",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
        "//compiler_gym/views",
    ],
)
