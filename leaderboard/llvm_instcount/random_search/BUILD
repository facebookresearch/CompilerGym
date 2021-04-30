# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "random_search",
    srcs = ["random_search.py"],
    deps = [
        "//compiler_gym",
    ],
)

py_test(
    name = "random_search_test",
    timeout = "short",
    srcs = ["random_search_test.py"],
    deps = [
        ":random_search",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)
