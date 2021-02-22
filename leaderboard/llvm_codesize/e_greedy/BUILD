# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "e_greedy",
    srcs = ["e_greedy.py"],
    deps = [
        "//leaderboard/llvm_codesize:eval_policy",
    ],
)

py_test(
    name = "e_greedy_test",
    timeout = "short",
    srcs = ["e_greedy_test.py"],
    deps = [
        ":e_greedy",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)
