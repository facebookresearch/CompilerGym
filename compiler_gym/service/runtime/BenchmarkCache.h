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

constexpr size_t kEvictionSizeInBytes = 256 * 1024 * 1024;

/**
 * A cache of Benchmark protocol messages.
 *
 * This object caches Benchmark messages by URI. Once the cache reaches a
 * predetermined size, benchmarks are evicted randomly until the capacity is
 * reduced to 50%.
 */
class BenchmarkCache {
 public:
  /**
   * Constructor.
   *
   * @param maxSizeInBytes The maximum size of the benchmark buffer before an
   *    automated eviction is run.
   * @param rand A random start used for selecting benchmarks for random
   *    eviction.
   */
  BenchmarkCache(size_t maxSizeInBytes = kEvictionSizeInBytes,
                 std::optional<std::mt19937_64> rand = std::nullopt);

  /**
   * Lookup a benchmark. The pointer set by this method is valid only until the
   * next call to add().
   *
   * @param uri The URI of the benchmark.
   * @return A Benchmark pointer.
   */
  const Benchmark* get(const std::string& uri) const;

  /**
   * Move-insert the given benchmark to the cache.
   *
   * @param benchmark A benchmark to insert.
   */
  void add(const Benchmark&& benchmark);

  /**
   * Get the number of elements in the cache.
   *
   * @return A nonnegative integer.
   */
  inline size_t size() const { return benchmarks_.size(); };

  /**
   * Get the size of the cache in bytes.
   *
   * @return A nonnegative integer.
   */
  inline size_t sizeInBytes() const { return sizeInBytes_; };

  /**
   * The maximum size of the cache before an eviction.
   *
   * @return A nonnegative integer.
   */
  inline size_t maxSizeInBytes() const { return maxSizeInBytes_; };

  /**
   * Set a new maximum size of the cache.
   *
   * @param maxSizeInBytes A number of bytes.
   */
  void setMaxSizeInBytes(size_t maxSizeInBytes);

  /**
   * Evict benchmarks randomly to reduce the capacity to the given size.
   *
   * If `targetSizeInBytes` is not provided, benchmarks are evicted to 50% of
   * `maxSizeInBytes`.
   *
   * @param targetSizeInBytes A target maximum size in bytes.
   */
  void evictToCapacity(std::optional<size_t> targetSizeInBytes = std::nullopt);

 private:
  std::unordered_map<std::string, const Benchmark> benchmarks_;

  std::mt19937_64 rand_;
  size_t maxSizeInBytes_;
  size_t sizeInBytes_;
};

}  // namespace compiler_gym::runtime
