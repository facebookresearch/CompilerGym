# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "leaderboard",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":llvm_instcount",
    ],
)

py_library(
    name = "llvm_instcount",
    srcs = ["llvm_instcount.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym:compiler_env_state",
        "//compiler_gym/bin:validate",
        "//compiler_gym/envs",
        "//compiler_gym/util",
    ],
)
