# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "actions_test",
    timeout = "long",
    srcs = ["actions_test.py"],
    shard_count = 1,
    deps = [
        "//compiler_gym",
        "//tests:test_main",
    ],
)
