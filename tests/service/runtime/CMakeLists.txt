# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_py_test(
  NAME
    benchmark_cache_test
  SRCS
    "benchmark_cache_test.py"
  DEPS
    compiler_gym::service::proto::proto
    compiler_gym::service::runtime::benchmark_cache
    tests::test_main
)

cg_cc_test(
  NAME
    BenchmarkCacheTest
  SRCS
    "BenchmarkCacheTest.cc"
  DEPS
    compiler_gym::service::proto::compiler_gym_service_cc
    compiler_gym::service::runtime::BenchmarkCache
    tests::TestMain
  ABS_DEPS
    GTest::GTest
)
