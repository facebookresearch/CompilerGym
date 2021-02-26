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
        "//compiler_gym/envs/llvm",
        "//compiler_gym/envs/llvm/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "cbench_test",
    timeout = "moderate",
    srcs = ["cbench_test.py"],
    deps = [
        "//compiler_gym/envs/llvm",
        "//compiler_gym/envs/llvm/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "cbench_validate_test",
    timeout = "eternal",
    srcs = ["cbench_validate_test.py"],
    shard_count = 8,
    deps = [
        "//compiler_gym",
        "//compiler_gym/datasets",
        "//compiler_gym/envs",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "github_test",
    timeout = "long",
    srcs = ["github_test.py"],
    deps = [
        "//compiler_gym/envs/llvm",
        "//compiler_gym/envs/llvm/datasets",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:llvm",
    ],
)
