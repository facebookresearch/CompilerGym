# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "benchmark_from_flags",
    srcs = ["benchmark_from_flags.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/service/proto",
    ],
)

py_library(
    name = "env_from_flags",
    srcs = ["env_from_flags.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
    ],
)

py_library(
    name = "nproc",
    srcs = ["nproc.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "output_dir",
    srcs = ["output_dir.py"],
    visibility = ["//visibility:public"],
)
