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

#include "compiler_gym/envs/llvm/service/BenchmarkDynamicConfig.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/StrLenConstexpr.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace fs = boost::filesystem;
namespace sys = boost::system;

using grpc::Status;
using grpc::StatusCode;

using BenchmarkProto = compiler_gym::Benchmark;
using BenchmarkDynamicConfigProto = compiler_gym::BenchmarkDynamicConfig;

namespace compiler_gym::llvm_service {

BenchmarkFactory::BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                                   std::optional<std::mt19937_64> rand,
                                   size_t maxLoadedBenchmarksCount)
    : workingDirectory_(workingDirectory),
      rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      maxLoadedBenchmarksCount_(maxLoadedBenchmarksCount) {
  CHECK(maxLoadedBenchmarksCount) << "Assertion maxLoadedBenchmarksCount > 0 failed! "
                                  << "maxLoadedBenchmarksCount = " << maxLoadedBenchmarksCount;
  VLOG(2) << "BenchmarkFactory initialized";
}

BenchmarkFactory::~BenchmarkFactory() { close(); }

void BenchmarkFactory::close() {
  VLOG(2) << "BenchmarkFactory closing with " << benchmarks_.size() << " entries";
  for (auto& entry : benchmarks_) {
    entry.second.close();
  }
  benchmarks_.clear();
}

Status BenchmarkFactory::getBenchmark(const BenchmarkProto& benchmarkMessage,
                                      std::unique_ptr<Benchmark>* benchmark) {
  // Check if the benchmark has already been loaded into memory.
  auto loaded = benchmarks_.find(benchmarkMessage.uri());
  if (loaded != benchmarks_.end()) {
    VLOG(3) << "LLVM benchmark cache hit: " << benchmarkMessage.uri();
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
          llvm::SmallString<0>(programFile.contents().begin(), programFile.contents().end()),
          benchmarkMessage.dynamic_config()));
      break;
    }
    case compiler_gym::File::DataCase::kUri: {
      VLOG(3) << "LLVM benchmark cache miss, read from URI: " << benchmarkMessage.uri();
      // Check the scheme of the benchmark URI.
      if (programFile.uri().find("file:///") != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid benchmark data URI. "
                                  "Only the file:/// scheme is supported: \"{}\"",
                                  programFile.uri()));
      }

      const fs::path path(programFile.uri().substr(util::strLen("file:///"), std::string::npos));
      RETURN_IF_ERROR(addBitcode(benchmarkMessage.uri(), path, benchmarkMessage.dynamic_config()));
      break;
    }
    case compiler_gym::File::DataCase::DATA_NOT_SET:
      return Status(StatusCode::INVALID_ARGUMENT, fmt::format("No program set in Benchmark:\n{}",
                                                              benchmarkMessage.DebugString()));
  }

  return getBenchmark(benchmarkMessage, benchmark);
}

Status BenchmarkFactory::addBitcode(const std::string& uri, const Bitcode& bitcode,
                                    std::optional<BenchmarkDynamicConfigProto> dynamicConfig) {
  Status status;
  std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module = makeModule(*context, bitcode, uri, &status);
  RETURN_IF_ERROR(status);
  DCHECK(module);

  if (benchmarks_.size() == maxLoadedBenchmarksCount_) {
    VLOG(2) << "LLVM benchmark cache reached maximum size " << maxLoadedBenchmarksCount_
            << ". Evicting random 50%.";
    for (int i = 0; i < static_cast<int>(maxLoadedBenchmarksCount_ / 2); ++i) {
      // Select a cached benchmark randomly.
      std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
      size_t index = distribution(rand_);
      auto iterator = std::next(std::begin(benchmarks_), index);

      // Evict the benchmark from the pool of loaded benchmarks.
      iterator->second.close();
      benchmarks_.erase(iterator);
    }
  }

  // TODO(cummins): This is very clumsy. In order to compute the baseline costs
  // we need a realized BenchmarkDynamicConfig. To create this, we need to
  // generate a scratch directory. This is then duplicated in the constructor of
  // the Benchmark class. Suggest a refactor.
  BenchmarkDynamicConfigProto realDynamicConfigProto =
      (dynamicConfig.has_value() ? *dynamicConfig : BenchmarkDynamicConfigProto());
  const fs::path scratchDirectory = createBenchmarkScratchDirectoryOrDie(workingDirectory_);
  BenchmarkDynamicConfig realDynamicConfig =
      realizeDynamicConfig(realDynamicConfigProto, scratchDirectory);

  BaselineCosts baselineCosts;
  RETURN_IF_ERROR(setBaselineCosts(*module, workingDirectory_, realDynamicConfig, &baselineCosts));

  sys::error_code ec;
  fs::remove_all(scratchDirectory, ec);
  if (ec) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to delete scratch directory: {}", scratchDirectory.string()));
  }

  benchmarks_.insert({uri, Benchmark(uri, std::move(context), std::move(module),
                                     realDynamicConfigProto, workingDirectory_, baselineCosts)});

  VLOG(2) << "Cached LLVM benchmark: " << uri << ". Cache size = " << benchmarks_.size()
          << " items";

  return Status::OK;
}

Status BenchmarkFactory::addBitcode(const std::string& uri, const fs::path& path,
                                    std::optional<BenchmarkDynamicConfigProto> dynamicConfig) {
  VLOG(2) << "addBitcode(" << path.string() << ")";

  Bitcode bitcode;
  RETURN_IF_ERROR(readBitcodeFile(path, &bitcode));
  return addBitcode(uri, bitcode, dynamicConfig);
}

}  // namespace compiler_gym::llvm_service
