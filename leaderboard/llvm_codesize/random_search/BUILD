# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_library(
    name = "random_search",
    srcs = ["random_search.py"],
    deps = [
        "//leaderboard/llvm_codesize:eval_policy",
    ],
)

py_test(
    name = "random_search_test",
    timeout = "short",
    srcs = ["random_search_test.py"],
    deps = [
        ":random_search",
        "//tests:test_main",
    ],
)
