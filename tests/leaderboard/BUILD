# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "llvm_instcount_test",
    srcs = ["llvm_instcount_test.py"],
    deps = [
        "//compiler_gym/leaderboard:llvm_instcount",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)
