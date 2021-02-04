# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

py_test(
    name = "bench_test",
    srcs = ["bench_test.py"],
    shard_count = 8,
    deps = [
        "//compiler_gym",
        "//compiler_gym/envs",
        "//compiler_gym/util",
        "//tests:test_main",
        "//tests/envs/llvm:fixtures",
    ],
)
