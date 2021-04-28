# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "example_compiler_gym_service",
    srcs = ["__init__.py"],
    data = [
        "//examples/example_compiler_gym_service/service_cc:compiler_gym-example-service-cc",
        "//examples/example_compiler_gym_service/service_py:compiler_gym-example-service-py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//compiler_gym/util",
    ],
)

py_test(
    name = "env_tests",
    srcs = ["env_tests.py"],
    deps = [
        ":example_compiler_gym_service",
        "//compiler_gym",
        "//tests:test_main",
    ],
)
