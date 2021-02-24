# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "connection_test",
    timeout = "short",
    srcs = ["connection_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/service",
        "//tests:test_main",
    ],
)
