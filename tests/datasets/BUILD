# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "benchmark_test",
    timeout = "short",
    srcs = ["benchmark_test.py"],
    deps = [
        "//compiler_gym/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "dataset_test",
    timeout = "short",
    srcs = ["dataset_test.py"],
    deps = [
        "//compiler_gym/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    deps = [
        "//compiler_gym/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "files_dataset_test",
    timeout = "short",
    srcs = ["files_dataset_test.py"],
    deps = [
        "//compiler_gym/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)
