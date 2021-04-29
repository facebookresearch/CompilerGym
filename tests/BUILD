# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_cc//cc:defs.bzl", "cc_library")

py_test(
    name = "compiler_env_test",
    srcs = ["compiler_env_test.py"],
    deps = [
        "//compiler_gym/datasets",
        "//compiler_gym/envs",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_test(
    name = "compiler_env_state_test",
    srcs = ["compiler_env_state_test.py"],
    deps = [
        "//compiler_gym",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_test(
    name = "random_search_test",
    timeout = "short",
    srcs = ["random_search_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym:random_replay",
        "//compiler_gym:random_search",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)

py_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.py"],
    visibility = [
        "//benchmarks:__subpackages__",
        "//examples:__subpackages__",
        "//leaderboard:__subpackages__",
        "//tests:__subpackages__",
    ],
    deps = [
        "//compiler_gym/util",
    ],
)

cc_library(
    name = "TestMacros",
    srcs = ["TestMacros.h"],
    visibility = [
        "//examples:__subpackages__",
        "//tests:__subpackages__",
    ],
)

cc_library(
    name = "TestMain",
    testonly = 1,
    srcs = ["TestMain.cc"],
    visibility = [
        "//examples:__subpackages__",
        "//tests:__subpackages__",
    ],
    deps = [
        "@glog",
        "@gtest",
    ],
)

py_test(
    name = "validate_test",
    srcs = ["validate_test.py"],
    deps = [
        ":test_main",
        "//compiler_gym",
    ],
)

py_test(
    name = "validation_result_test",
    srcs = ["validation_result_test.py"],
    deps = [
        ":test_main",
        "//compiler_gym",
    ],
)

py_test(
    name = "version_test",
    timeout = "short",
    srcs = ["version_test.py"],
    data = ["//:VERSION"],
    deps = [
        ":test_main",
        "//compiler_gym",
        "//tests/pytest_plugins:common",
    ],
)
