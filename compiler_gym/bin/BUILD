# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "bin",
    visibility = ["//visibility:public"],
    deps = [
        ":datasets",
        ":manual_env",
        ":random_replay",
        ":random_search",
        ":service",
        ":validate",
    ],
)

py_binary(
    name = "datasets",
    srcs = ["datasets.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":service",
        "//compiler_gym/datasets",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_binary(
    name = "manual_env",
    srcs = ["manual_env.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_binary(
    name = "random_eval",
    srcs = ["random_eval.py"],
    deps = [
        "//compiler_gym/util",
        "//compiler_gym/util/flags:output_dir",
    ],
)

py_binary(
    name = "random_search",
    srcs = ["random_search.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym:random_search",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
        "//compiler_gym/util/flags:nproc",
        "//compiler_gym/util/flags:output_dir",
    ],
)

py_binary(
    name = "random_replay",
    srcs = ["random_replay.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym:random_replay",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:env_from_flags",
        "//compiler_gym/util/flags:output_dir",
    ],
)

py_binary(
    name = "service",
    srcs = ["service.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_binary(
    name = "validate",
    srcs = ["validate.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym:validate",
        "//compiler_gym/util",
        "//compiler_gym/util/flags:env_from_flags",
        "//compiler_gym/util/flags:nproc",
    ],
)
