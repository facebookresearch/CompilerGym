# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "anghabench_test",
    timeout = "long",
    srcs = ["anghabench_test.py"],
    deps = [
        "//compiler_gym/envs/gcc",
        "//compiler_gym/envs/gcc/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:gcc",
    ],
)

py_test(
    name = "csmith_test",
    timeout = "long",
    srcs = ["csmith_test.py"],
    deps = [
        "//compiler_gym/envs/gcc",
        "//compiler_gym/envs/gcc/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:gcc",
    ],
)
