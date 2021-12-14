# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "loop_optimizations_service",
    srcs = ["__init__.py"],
    data = [
        "//examples/loop_optimizations_service/benchmarks",
        "//examples/loop_optimizations_service/service_py:loops-opt-service-py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/envs/llvm",
        "//compiler_gym/util",
    ],
)

py_test(
    name = "env_tests",
    srcs = ["env_tests.py"],
    deps = [
        ":loop_optimizations_service",
        "//compiler_gym",
        "//tests:test_main",
    ],
)

py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        ":loop_optimizations_service",
        "//compiler_gym",
    ],
)
