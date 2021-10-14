// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/runtime/BenchmarkCache.h"

#include <glog/logging.h>

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::runtime {

BenchmarkCache::BenchmarkCache(size_t maxSizeInBytes, std::optional<std::mt19937_64> rand)
    : rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      maxSizeInBytes_(maxSizeInBytes),
      sizeInBytes_(0){};

const Benchmark* BenchmarkCache::get(const std::string& uri) const {
  auto it = benchmarks_.find(uri);
  if (it == benchmarks_.end()) {
    return nullptr;
  }

  return &it->second;
}

void BenchmarkCache::add(const Benchmark&& benchmark) {
  const size_t benchmarkSize = benchmark.ByteSizeLong();

  // Remove any existing value to keep the cache size consistent.
  const auto it = benchmarks_.find(benchmark.uri());
  if (it != benchmarks_.end()) {
    const size_t replacedSize = it->second.ByteSizeLong();
    benchmarks_.erase(it);
    sizeInBytes_ -= replacedSize;
  }

  if (sizeInBytes() + benchmarkSize > maxSizeInBytes()) {
    if (benchmarkSize > maxSizeInBytes()) {
      LOG(WARNING) << "Adding new benchmark with size " << benchmarkSize
                   << " bytes exceeds total target cache size of " << maxSizeInBytes() << " bytes";
    } else {
      VLOG(3) << "Adding new benchmark with size " << benchmarkSize
              << " bytes exceeds maximum size " << maxSizeInBytes() << " bytes, " << this->size()
              << " items";
    }
    evictToCapacity();
  }

  benchmarks_.insert({benchmark.uri(), std::move(benchmark)});
  sizeInBytes_ += benchmarkSize;

  VLOG(3) << "Cached benchmark " << benchmark.uri() << " (" << benchmarkSize
          << " bytes). Cache size = " << sizeInBytes() << " bytes, " << size() << " items";
}

void BenchmarkCache::evictToCapacity(std::optional<size_t> targetSize) {
  int evicted = 0;
  targetSize = targetSize.has_value() ? targetSize : maxSizeInBytes() / 2;

  while (size() && sizeInBytes() > targetSize) {
    // Select a benchmark randomly.
    std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
    size_t index = distribution(rand_);
    auto iterator = std::next(std::begin(benchmarks_), index);

    // Evict the benchmark from the pool of loaded benchmarks.
    ++evicted;
    sizeInBytes_ -= iterator->second.ByteSizeLong();
    benchmarks_.erase(iterator);
  }

  if (evicted) {
    VLOG(2) << "Evicted " << evicted << " benchmarks from cache. Benchmark cache "
            << "size now " << sizeInBytes() << " bytes, " << benchmarks_.size() << " items";
  }
}

void BenchmarkCache::setMaxSizeInBytes(size_t maxSizeInBytes) {
  maxSizeInBytes_ = maxSizeInBytes;
  evictToCapacity(maxSizeInBytes);
}

}  // namespace compiler_gym::runtime
