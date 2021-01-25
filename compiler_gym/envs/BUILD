# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        "//compiler_gym/datasets:dataset",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
        "//compiler_gym/views",
    ],
)
