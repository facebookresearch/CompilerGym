# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "gcc_test",
    timeout = "long",
    srcs = ["gcc_test.py"],
    shard_count = 4,
    deps = [
        "//compiler_gym/envs",
        "//tests:test_main",
        "//tests/pytest_plugins:gcc",
    ],
)
