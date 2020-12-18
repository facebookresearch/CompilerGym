# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    visibility = ["//compiler_gym:__subpackages__"],
    deps = [
        "//compiler_gym/util",
    ],
)
