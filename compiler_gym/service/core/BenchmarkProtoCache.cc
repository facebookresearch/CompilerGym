// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/core/BenchmarkProtoCache.h"

#include <glog/logging.h>

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym {

namespace {

inline size_t getBenchmarkSize(const Benchmark& benchmark) { return sizeof(benchmark); }

}  // anonymous namespace

BenchmarkProtoCache::BenchmarkProtoCache(std::optional<std::mt19937_64> rand, size_t maxCacheSize)
    : rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      maxCacheSize_(maxCacheSize),
      cacheSize_(0){};

grpc::Status BenchmarkProtoCache::getBenchmark(const std::string& uri,
                                               const Benchmark** benchmark) const {
  auto it = benchmarks_.find(uri);
  if (it != benchmarks_.end()) {
    *benchmark = &it->second;
    return Status::OK;
  }

  return Status(StatusCode::NOT_FOUND, "Benchmark not found");
}

grpc::Status BenchmarkProtoCache::addBenchmark(const Benchmark&& benchmark) {
  const size_t benchmarkSize = getBenchmarkSize(benchmark);

  if (cacheSize() + benchmarkSize > maxCacheSize()) {
    VLOG(3) << "Adding new benchmark with size " << benchmarkSize
            << " exceeds maximum in-memory cache capacity " << maxCacheSize() << ", "
            << benchmarks_.size() << " protobufs";
    pruneBenchmarksCache();
  }

  benchmarks_.insert({benchmark.uri(), std::move(benchmark)});
  cacheSize() += benchmarkSize;

  VLOG(2) << "Added benchmark " << benchmark.uri() << ". Cache size = " << cacheSize() << " bytes, "
          << benchmarks_.size() << " entries";

  return Status::OK;
}

void BenchmarkProtoCache::pruneBenchmarksCache() {
  int evicted = 0;
  const size_t targetCapacity = maxCacheSize() / 2;
  while (benchmarks_.size() && cacheSize() > targetCapacity) {
    // Select a benchmark randomly.
    std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
    size_t index = distribution(rand_);
    auto iterator = std::next(std::begin(benchmarks_), index);

    // Evict the benchmark from the pool of loaded benchmarks.
    ++evicted;
    cacheSize() -= getBenchmarkSize(iterator->second);
    benchmarks_.erase(iterator);
  }

  VLOG(3) << "Evicted " << evicted << " benchmarks from cache. Benchmark cache "
          << "size now " << cacheSize() << " bytes, " << benchmarks_.size() << " benchmarks";
}

}  // namespace compiler_gym
