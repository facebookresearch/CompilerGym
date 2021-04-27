// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <string>

#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/StrLenConstexpr.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace fs = boost::filesystem;

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::llvm_service {

BenchmarkFactory::BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                                   std::optional<std::mt19937_64> rand,
                                   size_t maxLoadedBenchmarkSize)
    : workingDirectory_(workingDirectory),
      rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      loadedBenchmarksSize_(0),
      maxLoadedBenchmarkSize_(maxLoadedBenchmarkSize) {
  VLOG(2) << "BenchmarkFactory initialized";
}

Status BenchmarkFactory::getBenchmark(const std::string& uri,
                                      std::unique_ptr<Benchmark>* benchmark) {
  // Check if the benchmark has already been loaded into memory.
  auto loaded = benchmarks_.find(uri);
  if (loaded != benchmarks_.end()) {
    *benchmark = loaded->second.clone(workingDirectory_);
    return Status::OK;
  }

  return Status(StatusCode::NOT_FOUND, "Benchmark not found");
}

Status BenchmarkFactory::addBitcode(const std::string& uri, const Bitcode& bitcode) {
  Status status;
  std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module = makeModule(*context, bitcode, uri, &status);
  RETURN_IF_ERROR(status);
  DCHECK(module);

  const size_t bitcodeSize = bitcode.size();
  if (loadedBenchmarksSize_ + bitcodeSize > maxLoadedBenchmarkSize_) {
    VLOG(2) << "Adding new bitcode with size " << bitcodeSize
            << " exceeds maximum in-memory cache capacity " << maxLoadedBenchmarkSize_ << ", "
            << benchmarks_.size() << " bitcodes";
    int evicted = 0;
    // Evict benchmarks until we have reduced capacity below 50%.
    const size_t targetCapacity = maxLoadedBenchmarkSize_ / 2;
    while (benchmarks_.size() && loadedBenchmarksSize_ > targetCapacity) {
      // Select a cached benchmark randomly.
      std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
      size_t index = distribution(rand_);
      auto iterator = std::next(std::begin(benchmarks_), index);

      // Evict the benchmark from the pool of loaded benchmarks.
      ++evicted;
      loadedBenchmarksSize_ -= iterator->second.bitcodeSize();
      benchmarks_.erase(iterator);
    }

    VLOG(3) << "Evicted " << evicted << " benchmarks. Bitcode cache size now "
            << loadedBenchmarksSize_ << ", " << benchmarks_.size() << " bitcodes";
  }

  BaselineCosts baselineCosts;
  RETURN_IF_ERROR(setBaselineCosts(*module, &baselineCosts, workingDirectory_));

  benchmarks_.insert({uri, Benchmark(uri, std::move(context), std::move(module), bitcodeSize,
                                     workingDirectory_, baselineCosts)});
  loadedBenchmarksSize_ += bitcodeSize;

  return Status::OK;
}

Status BenchmarkFactory::addBitcode(const std::string& uri, const fs::path& path) {
  VLOG(2) << "addBitcode(" << path.string() << ")";

  Bitcode bitcode;
  RETURN_IF_ERROR(readBitcodeFile(path, &bitcode));
  return addBitcode(uri, bitcode);
}

}  // namespace compiler_gym::llvm_service
