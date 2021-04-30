# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "bench_test",
    srcs = ["bench_test.py"],
    shard_count = 8,
    deps = [
        "//compiler_gym",
        "//examples/example_compiler_gym_service",
        "//tests:test_main",
        "//tests/pytest_plugins:llvm",
    ],
)

py_binary(
    name = "parallelization_load_test",
    srcs = ["parallelization_load_test.py"],
    deps = [
        "//compiler_gym/util",
        "//compiler_gym/util/flags:benchmark_from_flags",
        "//compiler_gym/util/flags:env_from_flags",
    ],
)

py_test(
    name = "parallelization_load_test_test",
    timeout = "moderate",
    srcs = ["parallelization_load_test_test.py"],
    flaky = 1,
    deps = [
        ":parallelization_load_test",
        "//tests:test_main",
        "//tests/pytest_plugins:common",
        "//tests/pytest_plugins:llvm",
    ],
)
