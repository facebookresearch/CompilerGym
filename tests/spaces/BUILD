# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "commandline_test",
    timeout = "short",
    srcs = ["commandline_test.py"],
    deps = [
        "//compiler_gym/spaces",
        "//tests:test_main",
    ],
)

py_test(
    name = "named_discrete_test",
    timeout = "short",
    srcs = ["named_discrete_test.py"],
    deps = [
        "//compiler_gym/spaces",
        "//tests:test_main",
    ],
)

py_test(
    name = "scalar_test",
    timeout = "short",
    srcs = ["scalar_test.py"],
    deps = [
        "//compiler_gym/spaces",
        "//tests:test_main",
    ],
)

py_test(
    name = "sequence_test",
    timeout = "short",
    srcs = ["sequence_test.py"],
    deps = [
        "//compiler_gym/spaces",
        "//tests:test_main",
    ],
)
