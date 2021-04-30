# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "datasets",
    srcs = [
        "__init__.py",
        "anghabench.py",
        "cbench.py",
        "clgen.py",
        "csmith.py",
        "llvm_stress.py",
        "poj104.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs/llvm:llvm_benchmark",
        "//compiler_gym/service/proto",
        "//compiler_gym/third_party/llvm",
        "//compiler_gym/util",
    ],
)
