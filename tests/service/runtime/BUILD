# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "benchmark_cache_test",
    srcs = ["benchmark_cache_test.py"],
    deps = [
        "//compiler_gym/service/proto",
        "//compiler_gym/service/runtime:benchmark_cache",
        "//tests:test_main",
    ],
)

cc_test(
    name = "BenchmarkCacheTest",
    srcs = ["BenchmarkCacheTest.cc"],
    deps = [
        "//compiler_gym/service/proto:compiler_gym_service_cc",
        "//compiler_gym/service/runtime:BenchmarkCache",
        "//tests:TestMain",
        "@gtest",
    ],
)
