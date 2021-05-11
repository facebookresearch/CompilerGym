# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "views",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":observation",
        ":reward",
    ],
)

py_library(
    name = "observation",
    srcs = ["observation.py"],
    deps = [
        ":observation_space_spec",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "observation_space_spec",
    srcs = ["observation_space_spec.py"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "reward",
    srcs = ["reward.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":observation",
        "//compiler_gym/datasets",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
    ],
)
