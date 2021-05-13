// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <optional>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/service/runtime/BenchmarkCache.h"

using namespace ::testing;

namespace compiler_gym::runtime {
namespace {

// Test helper. Generate a benchmark of the given size in bytes.
Benchmark makeBenchmarkOfSize(const std::string& uri, int sizeInBytes, int target) {
  Benchmark bm;
  bm.set_uri(uri);
  std::vector<char> contents(target, '0');
  *bm.mutable_program()->mutable_contents() = {contents.begin(), contents.end()};
  int sizeOffset = bm.ByteSizeLong() - sizeInBytes;
  if (sizeOffset) {
    return makeBenchmarkOfSize(uri, sizeInBytes, sizeInBytes - sizeOffset);
  }

  return bm;
}

Benchmark makeBenchmarkOfSize(const std::string& uri, int sizeInBytes) {
  return makeBenchmarkOfSize(uri, sizeInBytes, sizeInBytes);
}

TEST(BenchmarkCache, makeBenchmarkOfSize) {
  // Sanity check for test helper function.
  ASSERT_EQ(makeBenchmarkOfSize("a", 10).ByteSizeLong(), 10);
  ASSERT_EQ(makeBenchmarkOfSize("abc", 10).ByteSizeLong(), 10);
  ASSERT_EQ(makeBenchmarkOfSize("a", 50).ByteSizeLong(), 50);
  ASSERT_EQ(makeBenchmarkOfSize("a", 100).ByteSizeLong(), 100);
  ASSERT_EQ(makeBenchmarkOfSize("a", 1024).ByteSizeLong(), 1024);
}

TEST(BenchmarkCache, replaceExistingItem) {
  BenchmarkCache cache;

  cache.add(makeBenchmarkOfSize("a", 30));
  ASSERT_EQ(cache.size(), 1);
  ASSERT_EQ(cache.sizeInBytes(), 30);

  cache.add(makeBenchmarkOfSize("a", 50));
  ASSERT_EQ(cache.size(), 1);
  ASSERT_EQ(cache.sizeInBytes(), 50);
}

TEST(BenchmarkCache, evictToCapacityOnMaxSizeReached) {
  BenchmarkCache cache;
  cache.setMaxSizeInBytes(100);

  cache.add(makeBenchmarkOfSize("a", 30));
  cache.add(makeBenchmarkOfSize("b", 30));
  cache.add(makeBenchmarkOfSize("c", 30));
  ASSERT_EQ(cache.sizeInBytes(), 90);
  ASSERT_EQ(cache.size(), 3);

  cache.add(makeBenchmarkOfSize("d", 30));
  ASSERT_EQ(cache.sizeInBytes(), 60);
  ASSERT_EQ(cache.size(), 2);
}

TEST(BenchmarkCache, getter) {
  BenchmarkCache cache;

  const auto a = makeBenchmarkOfSize("a", 30);
  cache.add(makeBenchmarkOfSize("a", 30));

  const auto b = makeBenchmarkOfSize("b", 50);
  cache.add(makeBenchmarkOfSize("b", 50));

  ASSERT_EQ(cache.get("a")->DebugString(), a.DebugString());
  ASSERT_NE(cache.get("a")->DebugString(), b.DebugString());
  ASSERT_EQ(cache.get("b")->DebugString(), b.DebugString());
}

TEST(BenchmarkCache, evictToCapacityOnMaximumSizeUpdate) {
  BenchmarkCache cache;

  cache.add(makeBenchmarkOfSize("a", 30));
  cache.add(makeBenchmarkOfSize("b", 30));
  cache.add(makeBenchmarkOfSize("c", 30));
  ASSERT_EQ(cache.sizeInBytes(), 90);
  ASSERT_EQ(cache.size(), 3);

  cache.setMaxSizeInBytes(50);
  ASSERT_EQ(cache.size(), 1);
  ASSERT_EQ(cache.sizeInBytes(), 30);
}

}  // anonymous namespace
}  // namespace compiler_gym::runtime
