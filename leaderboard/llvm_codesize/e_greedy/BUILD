# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "e_greedy",
    srcs = ["e_greedy.py"],
    deps = [
        "//compiler_gym/leaderboard:llvm_codesize",
    ],
)

py_test(
    name = "e_greedy_test",
    timeout = "short",
    srcs = ["e_greedy_test.py"],
    deps = [
        ":e_greedy",
        "//compiler_gym/leaderboard:llvm_codesize",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)
