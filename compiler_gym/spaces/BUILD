# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Extensions to OpenAI gym's spaces.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "spaces",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":box",
        ":commandline",
        ":common",
        ":dict",
        ":discrete",
        ":named_discrete",
        ":permutation",
        ":reward",
        ":scalar",
        ":sequence",
        ":space_sequence",
        ":tuple",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "box",
    srcs = ["box.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "commandline",
    srcs = ["commandline.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":named_discrete",
    ],
)

py_library(
    name = "dict",
    srcs = ["dict.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "discrete",
    srcs = ["discrete.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "named_discrete",
    srcs = ["named_discrete.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [":discrete"],
)

py_library(
    name = "reward",
    srcs = ["reward.py"],
    deps = [
        ":scalar",
        "//compiler_gym/util",
    ],
)

py_library(
    name = "scalar",
    srcs = ["scalar.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [":common"],
)

py_library(
    name = "sequence",
    srcs = ["sequence.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":common",
        ":scalar",
    ],
)

py_library(
    name = "space_sequence",
    srcs = ["space_sequence.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "tuple",
    srcs = ["tuple.py"],
    visibility = ["//compiler_gym:__subpackages__"],
)

py_library(
    name = "permutation",
    srcs = ["permutation.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        ":scalar",
        ":sequence",
    ],
)
