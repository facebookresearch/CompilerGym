# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "eval_policy",
    srcs = ["eval_policy.py"],
    visibility = ["//leaderboard/llvm_codesize:__subpackages__"],
    deps = [
        "//compiler_gym",
        "//compiler_gym/bin:validate",
    ],
)

py_test(
    name = "eval_policy_test",
    timeout = "short",
    srcs = ["eval_policy_test.py"],
    deps = [
        ":eval_policy",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
    ],
)
