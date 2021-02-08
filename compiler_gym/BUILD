# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This package contains CompilerGym, a toolkit for applying reinforcement
# learning to compilers.

py_library(
    name = "compiler_gym",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":random_search",
        ":validate",
        "//compiler_gym/envs",
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
    data = ["//compiler_gym/envs/llvm/service:compiler_gym-llvm-service"],
    visibility = ["//visibility:public"],
    deps = [
        ":random_replay",
        "//compiler_gym/envs",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "validate",
    srcs = ["validate.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/envs/llvm",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
    ],
)
