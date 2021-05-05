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

namespace compiler_gym {

constexpr size_t kMaxCacheSize = 512 * 1024 * 1024;

class BenchmarkProtoCache {
 public:
  BenchmarkProtoCache(std::optional<std::mt19937_64> rand = std::nullopt,
                      size_t maxCacheSize = kMaxCacheSize);

  // The pointer set by benchmark is valid only until the next call to
  // addBenchmark().
  [[nodiscard]] grpc::Status getBenchmark(const std::string& uri,
                                          const Benchmark** benchmark) const;

  [[nodiscard]] grpc::Status addBenchmark(const Benchmark&& benchmark);

  inline size_t cacheSize() const { return cacheSize_; };
  inline size_t maxCacheSize() const { return maxCacheSize_; };

 private:
  size_t& cacheSize() { return cacheSize_; };

  // Evict benchmarks randomly until we have reduced capacity below 50%.
  void pruneBenchmarksCache();

  std::unordered_map<std::string, const Benchmark> benchmarks_;

  std::mt19937_64 rand_;
  const size_t maxCacheSize_;
  size_t cacheSize_;
};

}  // namespace compiler_gym
