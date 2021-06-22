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

using BenchmarkProto = compiler_gym::Benchmark;

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

Status BenchmarkFactory::getBenchmark(const BenchmarkProto& benchmarkMessage,
                                      std::unique_ptr<Benchmark>* benchmark) {
  // Check if the benchmark has already been loaded into memory.
  auto loaded = benchmarks_.find(benchmarkMessage.uri());
  if (loaded != benchmarks_.end()) {
    VLOG(3) << "LLVM benchmark cache hit: " << benchmarkMessage.uri();
    ;
    *benchmark = loaded->second.clone(workingDirectory_);
    return Status::OK;
  }

  // Benchmark not cached, cache it and try again.
  const auto& programFile = benchmarkMessage.program();
  switch (programFile.data_case()) {
    case compiler_gym::File::DataCase::kContents: {
      VLOG(3) << "LLVM benchmark cache miss, add bitcode: " << benchmarkMessage.uri();
      RETURN_IF_ERROR(addBitcode(
          benchmarkMessage.uri(),
          llvm::SmallString<0>(programFile.contents().begin(), programFile.contents().end())));
      break;
    }
    case compiler_gym::File::DataCase::kUri: {
      VLOG(3) << "LLVM benchmark cache miss, read from URI: " << benchmarkMessage.uri();
      // Check the protocol of the benchmark URI.
      if (programFile.uri().find("file:///") != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid benchmark data URI. "
                                  "Only the file:/// protocol is supported: \"{}\"",
                                  programFile.uri()));
      }

      const fs::path path(programFile.uri().substr(util::strLen("file:///"), std::string::npos));
      RETURN_IF_ERROR(addBitcode(benchmarkMessage.uri(), path));
      break;
    }
    case compiler_gym::File::DataCase::DATA_NOT_SET:
      return Status(StatusCode::INVALID_ARGUMENT, fmt::format("No program set in Benchmark:\n{}",
                                                              benchmarkMessage.DebugString()));
  }

  return getBenchmark(benchmarkMessage, benchmark);
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
