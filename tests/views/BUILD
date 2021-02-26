# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "observation_test",
    srcs = ["observation_test.py"],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/views",
        "//tests:test_main",
    ],
)

py_test(
    name = "reward_test",
    srcs = ["reward_test.py"],
    deps = [
        "//compiler_gym/views",
        "//tests:test_main",
    ],
)
