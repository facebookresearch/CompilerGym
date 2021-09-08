# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "gcc",
    srcs = [
        "__init__.py",
        "gcc.py",
        "gcc_env.py",
    ],
    data = [
        "//compiler_gym/envs/gcc/service",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs:compiler_env",
        "//compiler_gym/envs/gcc/datasets",
        "//compiler_gym/service",
        "//compiler_gym/service/runtime",  # Implicit dependency of service.
        "//compiler_gym/util",
    ],
)
