# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This package contains CompilerGym, a toolkit for applying reinforcement
# learning to compilers.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "compiler_gym",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":random_search",
        ":validate",
        "//compiler_gym/envs",
        "//compiler_gym/leaderboard",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "compiler_env_state",
    srcs = ["compiler_env_state.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        "//compiler_gym/datasets:uri",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "random_replay",
    srcs = ["random_replay.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "random_search",
    srcs = ["random_search.py"],
    data = ["//compiler_gym/envs/llvm/service"],
    visibility = ["//visibility:public"],
    deps = [
        ":random_replay",
        "//compiler_gym/envs",
        "//compiler_gym/service:connection",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "validate",
    srcs = ["validate.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":validation_error",
        ":validation_result",
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "validation_error",
    srcs = ["validation_error.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "validation_result",
    srcs = ["validation_result.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":compiler_env_state",
        ":validation_error",
        "//compiler_gym/util",
    ],
)
