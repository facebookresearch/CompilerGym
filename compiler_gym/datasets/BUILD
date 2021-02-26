# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
    ],
)

py_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":benchmark",
        "//compiler_gym/util",
    ],
)
