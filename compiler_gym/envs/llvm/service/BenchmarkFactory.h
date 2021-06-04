// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <array>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

/**
 * Maximum number of bytes before benchmark cache eviction.
 *
 * Benchmarks are loaded from disk and cached in-memory so that future uses do
 * not require a disk access. The number of benchmarks that may be
 * simultaneously loaded is limited by the combined size of the bitcodes, in
 * bytes. Once this size is reached, benchmarks are offloaded so that they must
 * be re-read from disk.
 */
constexpr size_t kMaxLoadedBenchmarkSize = 512 * 1024 * 1024;

/**
 * A factory object for instantiating LLVM modules for use in optimization
 * sessions.
 *
 * Example usage:
 *
 * \code{.cpp}
 *     BenchmarkFactory factory;
 *     auto benchmark = factory.getBenchmark("file:////tmp/my_bitcode.bc");
 *     // ... do fun stuff
 * \endcode
 */
class BenchmarkFactory {
 public:
  /**
   * Return the global benchmark factory singleton.
   *
   * @param workingDirectory The working directory.
   * @param rand An optional random number generator. This is used for cache
   *     evictions.
   * @param maxLoadedBenchmarkSize The maximum size in bytes of the benchmark
   *     cache before evictions.
   * @return The benchmark factory singleton instance.
   */
  static BenchmarkFactory& getSingleton(const boost::filesystem::path& workingDirectory,
                                        std::optional<std::mt19937_64> rand = std::nullopt,
                                        size_t maxLoadedBenchmarkSize = kMaxLoadedBenchmarkSize) {
    static BenchmarkFactory instance(workingDirectory, rand, maxLoadedBenchmarkSize);
    return instance;
  }

  /**
   * Get the requested named benchmark.
   *
   * @param benchmarkMessage A Benchmark protocol message.
   * @param benchmark A benchmark instance to assign this benchmark to.
   * @return `OK` on success, or `INVALID_ARGUMENT` if the protocol message is
   *    invalid.
   */
  [[nodiscard]] grpc::Status getBenchmark(const compiler_gym::Benchmark& benchmarkMessage,
                                          std::unique_ptr<Benchmark>* benchmark);

 private:
  [[nodiscard]] grpc::Status addBitcode(const std::string& uri, const Bitcode& bitcode);

  [[nodiscard]] grpc::Status addBitcode(const std::string& uri,
                                        const boost::filesystem::path& path);

  /**
   * Construct a benchmark factory.
   *
   * @param workingDirectory A filesystem directory to use for storing temporary
   *    files.
   * @param rand is a random seed used to control the selection of random
   *    benchmarks.
   * @param maxLoadedBenchmarkSize is the maximum combined size of the bitcodes
   *    that may be cached in memory. Once this size is reached, benchmarks are
   *    offloaded so that they must be re-read from disk.
   */
  BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                   std::optional<std::mt19937_64> rand = std::nullopt,
                   size_t maxLoadedBenchmarkSize = kMaxLoadedBenchmarkSize);

  BenchmarkFactory(const BenchmarkFactory&) = delete;
  BenchmarkFactory& operator=(const BenchmarkFactory&) = delete;

  /**
   * A mapping from URI to benchmarks which have been loaded into memory.
   */
  std::unordered_map<std::string, Benchmark> benchmarks_;

  const boost::filesystem::path workingDirectory_;
  std::mt19937_64 rand_;
  /**
   * The current and maximum allowed sizes of the loaded benchmarks.
   */
  size_t loadedBenchmarksSize_;
  const size_t maxLoadedBenchmarkSize_;
};

}  // namespace compiler_gym::llvm_service