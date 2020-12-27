# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "views",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":observation",
        ":reward",
        ":reward_space_spec",
    ],
)

py_library(
    name = "observation",
    srcs = ["observation.py"],
    deps = [
        ":observation_space_spec",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
    ],
)

py_library(
    name = "observation_space_spec",
    srcs = ["observation_space_spec.py"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
        "//compiler_gym/spaces",
    ],
)

py_library(
    name = "reward",
    srcs = ["reward.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":reward_space_spec",
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
    ],
)

py_library(
    name = "reward_space_spec",
    srcs = ["reward_space_spec.py"],
    deps = [
        "//compiler_gym/service",
        "//compiler_gym/service/proto",
    ],
)
