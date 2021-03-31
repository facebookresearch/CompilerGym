# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "tabular_q_eval",
    srcs = ["tabular_q_eval.py"],
    deps = [
        "//compiler_gym/leaderboard:llvm_instcount",
        "//examples:tabular_q",
    ],
)
