# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "commandline_wrappers_test",
    timeout = "short",
    srcs = ["commandline_wrappers_test.py"],
    deps = [
        "//compiler_gym/wrappers",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "core_wrappers_test",
    timeout = "moderate",
    srcs = ["core_wrappers_test.py"],
    deps = [
        "//compiler_gym/wrappers",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "datasets_wrappers_test",
    timeout = "short",
    srcs = ["datasets_wrappers_test.py"],
    deps = [
        "//compiler_gym/wrappers",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "locked_step_test",
    timeout = "short",
    srcs = ["locked_step_test.py"],
    deps = [
        "//compiler_gym/wrappers",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "time_limit_wrappers_test",
    timeout = "short",
    srcs = ["time_limit_wrappers_test.py"],
    deps = [
        "//compiler_gym/wrappers",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)
