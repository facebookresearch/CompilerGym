// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>
#include <optional>
#include <random>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::runtime {

constexpr size_t kEvictionSizeInBytes = 512 * 1024 * 1024;

// An in-memory cache of Benchmark protocol buffers.
//
// This object caches Benchmark messages by URI. Once the cache reaches a
// predetermined size, benchmarks are evicted randomly until the capacity is
// reduced to 50%.
class BenchmarkCache {
 public:
  BenchmarkCache(size_t maxSizeInBytes = kEvictionSizeInBytes,
                 std::optional<std::mt19937_64> rand = std::nullopt);

  // The pointer set by benchmark is valid only until the next call to add().
  const Benchmark* get(const std::string& uri) const;

  // Move-insert the given benchmark to the cache.
  void add(const Benchmark&& benchmark);

  inline size_t size() const { return benchmarks_.size(); };
  inline size_t sizeInBytes() const { return sizeInBytes_; };
  inline size_t maxSizeInBytes() const { return maxSizeInBytes_; };

  void setMaxSizeInBytes(size_t maxSizeInBytes);

  // Evict benchmarks randomly to reduce the capacity to the given size. If
  // targetSizeInBytes is not provided, benchmarks are evicted to 50% of
  // maxSizeInBytes.
  void evictToCapacity(std::optional<size_t> targetSizeInBytes = std::nullopt);

 private:
  std::unordered_map<std::string, const Benchmark> benchmarks_;

  std::mt19937_64 rand_;
  size_t maxSizeInBytes_;
  size_t sizeInBytes_;
};

}  // namespace compiler_gym::runtime
