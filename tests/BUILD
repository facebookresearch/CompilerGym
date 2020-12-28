# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_test(
    name = "random_search_test",
    timeout = "short",
    srcs = ["random_search_test.py"],
    deps = [
        "//compiler_gym",
        "//compiler_gym:random_replay",
        "//compiler_gym:random_search",
        "//tests:test_main",
    ],
)

py_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.py"],
    visibility = [
        "//examples:__subpackages__",
        "//tests:__subpackages__",
    ],
    deps = [
        "//compiler_gym",
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
    name = "version_test",
    srcs = ["version_test.py"],
    deps = [
        ":test_main",
        "//compiler_gym",
    ],
)
