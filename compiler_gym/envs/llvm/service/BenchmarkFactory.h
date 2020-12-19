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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

// Benchmarks are loaded from disk and cached in-memory so that future uses
// do not require a disk access. The number of benchmarks that may be
// simultaneously loaded is limited by the combined size of the bitcodes, in
// bytes. Once this size is reached, benchmarks are offloaded so that they must
// be re-read from disk.
constexpr size_t kMaxLoadedBenchmarkSize = 512 * 1024 * 1024;

// A factory object for instantiating LLVM modules for use in optimization
// sessions. Example usage:
//
//     BenchmarkFactory factory;
//     for (int i = 0; i < 10; ++i) {
//       auto benchmark = factory.getBenchmark();
//       // ... do fun stuff
//     }
class BenchmarkFactory {
 public:
  // Construct a benchmark factory. rand is a random seed used to control the
  // selection of random benchmarks. maxLoadedBenchmarkSize is the maximum
  // combined size of the bitcodes that may be cached in memory. Once this
  // size is reached, benchmarks are offloaded so that they must be re-read from
  // disk.
  BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                   std::optional<std::mt19937_64> rand = std::nullopt,
                   size_t maxLoadedBenchmarkSize = kMaxLoadedBenchmarkSize);

  // Add a new bitcode. bitcodePath is optional. If provided, it allows the
  // newly added benchmark to be evicted from the in-memory cache.
  [[nodiscard]] grpc::Status addBitcode(
      const std::string& uri, const Bitcode& bitcode,
      std::optional<boost::filesystem::path> bitcodePath = std::nullopt);

  // Add a bitcode URI alias. For example,
  //    addBitcodeFile("benchmark://foo", "file:///tmp/foo.bc")
  // adds a new benchmark "benchmark://foo" which resolves to the path
  // "/tmp/foo.bc".
  [[nodiscard]] grpc::Status addBitcodeUriAlias(const std::string& src, const std::string& dst);

  // Add a directory of bitcode files. The format for added benchmark URIs is
  // `benchmark://<relStem>`, where relStem is the path of the file, relative
  // to the root of the directory, without the file extension.
  //
  // Note that if any of the bitcodes are invalid, this error will be latent
  // until a call to getBenchmark() attempts to load it.
  [[nodiscard]] grpc::Status addDirectoryOfBitcodes(const boost::filesystem::path& path);

  // Get a random benchmark.
  [[nodiscard]] grpc::Status getBenchmark(std::unique_ptr<Benchmark>* benchmark);

  // Get the requested named benchmark.
  [[nodiscard]] grpc::Status getBenchmark(const std::string& uri,
                                          std::unique_ptr<Benchmark>* benchmark);

  // Enumerate the list of available benchmark names that can be
  // passed to getBenchmark().
  [[nodiscard]] std::vector<std::string> getBenchmarkNames() const;

  // Scan the site data directory for new files. This is used to indicate that
  // the directory has changed.
  [[nodiscard]] grpc::Status scanSiteDataDirectory();

  size_t numBenchmarks() const;

 private:
  // Add a directory of bitcode files by reading a MANIFEST file. The manifest
  // file must consist of a single relative path per line.
  [[nodiscard]] grpc::Status addDirectoryOfBitcodes(const boost::filesystem::path& path,
                                                    const boost::filesystem::path& manifestPath);

  // Fetch a random benchmark matching a given URI prefix.
  [[nodiscard]] grpc::Status getBenchmarkByUriPrefix(const std::string& uriPrefix,
                                                     const std::string& resolvedUriPrefix,
                                                     std::unique_ptr<Benchmark>* benchmark);

  [[nodiscard]] grpc::Status addBitcodeFile(const std::string& uri,
                                            const boost::filesystem::path& path);

  [[nodiscard]] grpc::Status loadBenchmark(
      std::unordered_map<std::string, boost::filesystem::path>::const_iterator iterator,
      std::unique_ptr<Benchmark>* benchmark);

  // A map from benchmark name to the path of a bitcode file. This is used to
  // store the paths of benchmarks w
  // hich have not yet been loaded into memory.
  // Once loaded, they are removed from this map and replaced by an entry in
  // benchmarks_.
  std::unordered_map<std::string, boost::filesystem::path> unloadedBitcodePaths_;
  // A mapping from URI to benchmarks which have been loaded into memory.
  std::unordered_map<std::string, Benchmark> benchmarks_;

  const boost::filesystem::path workingDirectory_;
  std::mt19937_64 rand_;
  // The current and maximum allowed sizes of the loaded benchmarks.
  size_t loadedBenchmarksSize_;
  const size_t maxLoadedBenchmarkSize_;
};

}  // namespace compiler_gym::llvm_service
