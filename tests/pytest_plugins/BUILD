# Plugins and test fixtures for CompilerGym.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "llvm",
    testonly = 1,
    srcs = ["llvm.py"],
    data = [
        "//compiler_gym/envs/llvm/service/passes:actions_genfiles",
        "//compiler_gym/third_party/cBench:benchmarks_list",
    ],
    deps = [
        "//compiler_gym",
        "//compiler_gym/third_party/llvm",
    ],
)

py_library(
    name = "common",
    testonly = 1,
    srcs = ["common.py"],
)

py_library(
    name = "random_util",
    testonly = 1,
    srcs = ["random_util.py"],
    deps = [
        "//compiler_gym",
    ],
)
